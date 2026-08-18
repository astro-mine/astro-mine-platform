#!/usr/bin/env bash
# Stand up the ephemeral cluster the opt-in `cluster`-marked tests run against (cloud.md §10).
#
#   ./platform/kind/up.sh                 # engines + Kueue + MinIO (the default)
#   ADMISSION=1 ./platform/kind/up.sh     # ...plus Kyverno + the cosign admission policy
#
# It writes ./platform/kind/harness.env; source it, then run the tests:
#
#   set -a && . ./platform/kind/harness.env && set +a
#   uv run pytest -m cluster
#
# Idempotent: re-running against an existing cluster upgrades it in place. Tear down with
# ./platform/kind/down.sh (which is what the CI job's always() step calls).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-astro-mine}"
REGISTRY_NAME="${REGISTRY_NAME:-kind-registry}"
REGISTRY="${REGISTRY_NAME}:5000"
# The self-signed CA + server cert registry.sh mints for the registry's TLS (astro-mine-cloud#30).
# up.sh trusts it in three places: the nodes' containerd, the Kyverno admission controller, and the
# crane container that pushes the workload image. Same default as registry.sh (both live here).
REGISTRY_CERT_DIR="${REGISTRY_CERT_DIR:-${HERE}/registry-certs}"
RELEASE="${RELEASE:-astro-mine}"
CHART="${ROOT}/platform/helm/astro-mine-cloud"
NAMESPACE="astro-mine-system"
IMAGE_REPO="${REGISTRY}/astro-mine-workload"
IMAGE_TAG="${IMAGE_TAG:-test}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-${HERE}/kubeconfig}"
ADMISSION="${ADMISSION:-0}"
INSTALL_RAY="${INSTALL_RAY:-1}"
COSIGN_KEYDIR="${COSIGN_KEYDIR:-${HERE}/cosign}"

log() { printf '\n=== %s\n' "$*"; }

for binary in docker kind kubectl helm; do
  command -v "${binary}" >/dev/null || { echo "missing required binary: ${binary}" >&2; exit 1; }
done

# --- 1. registry + cluster -------------------------------------------------------------------

log "local registry"
bash "${HERE}/registry.sh"

log "kind cluster ${CLUSTER_NAME}"
if kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
  echo "cluster ${CLUSTER_NAME} already exists"
else
  # --retain keeps the node containers alive when create fails. Without it kind deletes them on the
  # way out and takes containerd's and kubelet's logs with it -- leaving only kubeadm's "context
  # deadline exceeded", which says the control plane never came up but never says why. On failure,
  # dump what the nodes actually recorded and *then* fail.
  if ! kind create cluster \
        --name "${CLUSTER_NAME}" \
        --config "${HERE}/cluster.yaml" \
        --wait 180s \
        --retain; then
    echo "::group::kind: node logs (cluster creation FAILED)"
    kind export logs "${LOG_DIR:-/tmp/kind-logs}" --name "${CLUSTER_NAME}" || true
    for node in $(kind get nodes --name "${CLUSTER_NAME}" 2>/dev/null); do
      echo "===== ${node}: containerd"
      docker exec "${node}" journalctl -u containerd --no-pager -n 120 2>/dev/null || true
      echo "===== ${node}: kubelet"
      docker exec "${node}" journalctl -u kubelet --no-pager -n 120 2>/dev/null || true
      echo "===== ${node}: containerd config"
      docker exec "${node}" containerd config dump 2>/dev/null | head -40 || true
    done
    echo "::endgroup::"
    kind delete cluster --name "${CLUSTER_NAME}" || true
    exit 1
  fi
fi
kind get kubeconfig --name "${CLUSTER_NAME}" > "${KUBECONFIG_PATH}"
export KUBECONFIG="${KUBECONFIG_PATH}"

# The registry must sit on the kind network for the nodes' containerd to resolve `kind-registry`
# by name (docker's embedded DNS). The `kind` network only exists once a cluster does, so this is
# necessarily after the create.
if [ "$(docker inspect -f '{{json .NetworkSettings.Networks.kind}}' "${REGISTRY_NAME}")" = 'null' ]; then
  docker network connect kind "${REGISTRY_NAME}"
fi

# Point each node's containerd at the local registry over TLS, trusting the harness CA.
#
# containerd 2.x dropped the `registry.mirrors` / `registry.configs` config blocks; the only
# supported way to describe a registry is a hosts.toml under the `config_path` set in cluster.yaml.
# The registry serves HTTPS (registry.sh), so the endpoint is `https://` and `ca` names the CA that
# signed its cert -- copied onto each node next to the hosts.toml. (This replaces the earlier
# plain-HTTP endpoint that skipped TLS verification entirely; see astro-mine-cloud#30.)
log "registry hosts.toml on each node"
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  docker exec "${node}" mkdir -p "/etc/containerd/certs.d/${REGISTRY}"
  docker cp "${REGISTRY_CERT_DIR}/ca.crt" "${node}:/etc/containerd/certs.d/${REGISTRY}/ca.crt"
  docker exec -i "${node}" cp /dev/stdin "/etc/containerd/certs.d/${REGISTRY}/hosts.toml" <<EOF
[host."https://${REGISTRY}"]
  capabilities = ["pull", "resolve", "push"]
  ca = "/etc/containerd/certs.d/${REGISTRY}/ca.crt"
EOF
done

# The documented "local registry hosting" contract, so tooling can discover the registry.
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:5001"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

# --- 2. the platform chart, for real ----------------------------------------------------------
#
# `helm dependency build` + `helm install` -- not the `helm lint` the default CI job runs. Linting
# renders templates; it does not tell you whether KubeRay, Kueue and Argo actually come up, which
# is the thing RM-P1-CLOUD-01 claims.

log "helm dependency build"
helm dependency build "${CHART}"

PROFILE="${ROOT}/platform/profiles/kind.yaml"
if [ "${ADMISSION}" = "1" ]; then
  PROFILE="${ROOT}/platform/profiles/kind-admission.yaml"
fi

log "helm install (profile: $(basename "${PROFILE}"))"
# Phase 1: bring the operators up with admission OFF even when it is wanted. Kyverno's own
# validating webhook has to be live before it will accept a ClusterPolicy, and its CRDs have to
# exist before Helm can render one -- installing both in a single pass is a race we would lose
# intermittently, which is worse than losing it every time.
helm upgrade --install "${RELEASE}" "${CHART}" \
  -f "${PROFILE}" \
  --set admission.enabled=false \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --wait --timeout 15m

log "MinIO + namespaces"
kubectl apply -f "${HERE}/minio.yaml"
kubectl -n "${NAMESPACE}" rollout status deployment/minio --timeout=300s
kubectl -n "${NAMESPACE}" wait --for=condition=Complete job/minio-create-bucket --timeout=300s

# --- 3. the workload image --------------------------------------------------------------------

# The version the image must report. RunContext.code_version is *inside* the content address, so an
# image whose platform version differs from the host's produces a different address and fails the
# equivalence test for a reason that has nothing to do with the run.
#
# It used to be *injected*: hatch-vcs derived the version from `git describe`, the build context has
# no .git, so the host's answer was passed in as SETUPTOOLS_SCM_PRETEND_VERSION. Consolidation moved
# the build to maturin and pinned `version = "0.1.0"` statically, so the two now agree by
# construction and there is nothing to inject. The host's answer is still read and still passed --
# as something for the image to be *checked against*, which is a weaker claim that stays true.
log "workload image"
CODE_VERSION="$(uv run python -c 'import astro_mine.cloud as c; print(c.__version__)')"
echo "asserting the image reports the host's platform version: ${CODE_VERSION}"

# The `core_token` build secret is gone with the repositories it authenticated to: the platform is
# one package with no private git sources, so the image builds from this checkout and reaches only
# the public indexes.
DOCKER_BUILDKIT=1 docker build \
  --file "${HERE}/workload.Dockerfile" \
  --build-arg "ASTRO_MINE_PLATFORM_VERSION=${CODE_VERSION}" \
  --build-arg "INSTALL_RAY=${INSTALL_RAY}" \
  --tag "${IMAGE_REPO}:${IMAGE_TAG}" \
  "${ROOT}"

# Push from a container on the kind network. The host's docker daemon cannot push to
# `kind-registry:5000` (it would insist on TLS for a non-loopback name, and teaching it otherwise
# means editing /etc/docker/daemon.json and restarting the daemon); crane, run *inside* the
# network, resolves the registry exactly as the nodes and Kyverno do. So there is one image
# reference in the world and everyone can resolve it.
SAVED_DIR="$(mktemp -d)"
SAVED="${SAVED_DIR}/image.tar"
docker save "${IMAGE_REPO}:${IMAGE_TAG}" -o "${SAVED}"

# The crane image runs as nonroot (uid 65532), but `mktemp -d` gives 0700 owned by the runner --
# so the container cannot even stat the bind-mounted tar. Open the directory and the tar up rather
# than running crane as root: it only ever reads them, and this keeps crane unprivileged.
chmod 0755 "${SAVED_DIR}"
chmod 0644 "${SAVED}"

# crane over HTTPS, trusting the harness CA (SSL_CERT_FILE, which Go's cert pool honours). The
# registry serves TLS now, and crane's `--insecure` would force *plaintext* HTTP -- which the TLS
# registry refuses (astro-mine-cloud#30) -- so it is gone; the CA mount is what makes the push work.
docker run --rm --network kind \
  -v "${SAVED_DIR}:/img:ro" \
  -v "${REGISTRY_CERT_DIR}/ca.crt:/certs/ca.crt:ro" \
  -e SSL_CERT_FILE=/certs/ca.crt \
  gcr.io/go-containerregistry/crane:v0.21.7 \
  push /img/image.tar "${IMAGE_REPO}:${IMAGE_TAG}"

IMAGE_DIGEST="$(docker run --rm --network kind \
  -v "${REGISTRY_CERT_DIR}/ca.crt:/certs/ca.crt:ro" \
  -e SSL_CERT_FILE=/certs/ca.crt \
  gcr.io/go-containerregistry/crane:v0.21.7 \
  digest "${IMAGE_REPO}:${IMAGE_TAG}")"
WORKLOAD_IMAGE="${IMAGE_REPO}@${IMAGE_DIGEST}"
echo "workload image: ${WORKLOAD_IMAGE}"

# --- 4. admission (opt-in) --------------------------------------------------------------------

if [ "${ADMISSION}" = "1" ]; then
  log "kyverno + the cosign admission policy"
  # Kyverno is a SUBCHART of the umbrella release, so Helm puts it in the release namespace -- there
  # is no `kyverno` namespace to wait in. Wait on every deployment in that namespace rather than
  # selecting Kyverno's by label: the chart's labels are its business, not ours, and by this point
  # KubeRay, Kueue, Argo and MinIO all have to be Available anyway. `--all` says exactly that and
  # cannot silently match nothing, which is how a label selector fails.
  kubectl -n "${NAMESPACE}" wait --for=condition=Available deployment --all --timeout=300s

  # Kyverno fetches signatures from the registry from *inside its own pod*, so it needs to resolve
  # `kind-registry`. A pod's DNS search path is <its-namespace>.svc.cluster.local first, so the
  # Service must live in the SAME namespace as the Kyverno pod -- the release namespace -- and carry
  # that exact name. It has no selector: the backend is a docker container, not a pod, so the
  # endpoints are set by hand to its IP on the kind network (which pods can route to).
  REGISTRY_IP="$(docker inspect -f '{{.NetworkSettings.Networks.kind.IPAddress}}' "${REGISTRY_NAME}")"
  kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${REGISTRY_NAME}
  namespace: ${NAMESPACE}
spec:
  ports:
    - name: registry
      port: 5000
      targetPort: 5000
---
apiVersion: v1
kind: Endpoints
metadata:
  name: ${REGISTRY_NAME}
  namespace: ${NAMESPACE}
subsets:
  - addresses:
      - ip: ${REGISTRY_IP}
    ports:
      - name: registry
        port: 5000
EOF

  # The key pair the admission test signs with. Generated here so the chart is installed with the
  # key it will actually verify against -- and *exported* below, so the test signs with that key
  # rather than minting its own. It did mint its own, and Kyverno duly refused the "signed" image:
  # it was signed by a stranger. The public half in the policy and the private half in the test have
  # to be two halves of one pair, or the test asserts nothing it means to.
  rm -rf "${COSIGN_KEYDIR}" && mkdir -p "${COSIGN_KEYDIR}"
  ( cd "${COSIGN_KEYDIR}" && COSIGN_PASSWORD="" cosign generate-key-pair >/dev/null )
  KEYDIR="${COSIGN_KEYDIR}"

  # Phase 2: now that Kyverno is live, apply the policy -- and hand Kyverno the registry CA so its
  # signature *fetch* trusts the harness registry's TLS cert. Without it Kyverno reaches the registry
  # over HTTPS (it always does, even with allowInsecure) but cannot verify a self-signed cert, and
  # refuses every image for a transport reason (astro-mine-cloud#30). The chart mounts this over
  # /etc/ssl/certs/ca-certificates.crt -- the file Kyverno's Go registry client reads.
  helm upgrade "${RELEASE}" "${CHART}" \
    -f "${PROFILE}" \
    --set-file "admission.cosignPublicKey=${KEYDIR}/cosign.pub" \
    --set-file "kyverno.admissionController.caCertificates.data=${REGISTRY_CERT_DIR}/ca.crt" \
    --namespace "${NAMESPACE}" \
    --wait --timeout 10m

  kubectl wait --for=condition=Ready clusterpolicy/require-signed-images --timeout=180s
fi

# --- 5. the environment the tests read --------------------------------------------------------

cat > "${HERE}/harness.env" <<EOF
# Written by platform/kind/up.sh -- source it before \`uv run pytest -m cluster\`:
#   set -a && . ./platform/kind/harness.env && set +a
ASTRO_MINE_CLUSTER_KUBECONFIG=${KUBECONFIG_PATH}
ASTRO_MINE_WORKLOAD_IMAGE=${WORKLOAD_IMAGE}
ASTRO_MINE_CLUSTER_REGISTRY=${REGISTRY}
# The self-signed CA the harness registry serves TLS with. The admission test's crane trusts it
# (SSL_CERT_FILE) so it can push the signed/unsigned images over HTTPS (astro-mine-cloud#30).
ASTRO_MINE_REGISTRY_CA=${REGISTRY_CERT_DIR}/ca.crt
ASTRO_MINE_S3_BUCKET=astro-mine
# ...as a pod reaches the store (cluster DNS) vs as the host does (kind's node-port mapping).
ASTRO_MINE_S3_ENDPOINT=http://minio.astro-mine-system.svc.cluster.local:9000
ASTRO_MINE_S3_ENDPOINT_HOST=http://localhost:30900
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_DEFAULT_REGION=us-east-1
ASTRO_MINE_ADMISSION=${ADMISSION}
# The key pair whose public half is in the Kyverno policy. The admission test MUST sign with this
# one -- a freshly minted key is a key the cluster has never seen, and Kyverno rejects the "signed"
# image exactly as it should, turning the admit test into a second reject test.
ASTRO_MINE_COSIGN_KEYDIR=${COSIGN_KEYDIR}
EOF

log "ready"
cat "${HERE}/harness.env"
