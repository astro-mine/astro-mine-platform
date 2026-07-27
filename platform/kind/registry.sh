#!/usr/bin/env bash
# The local OCI registry the workload image is pushed to, on the kind docker network.
#
# Why not just `kind load docker-image`? Because the admission tests need a *registry*: Kyverno
# fetches cosign signatures and attestations from one, and it does so from inside its own pod. So
# there has to be a registry the cluster itself can reach, and the image reference has to name it.
#
# `kind-registry:5000` is that name. The nodes resolve it through docker's embedded DNS (both are
# on the `kind` network); Kyverno resolves it through the Service/Endpoints `up.sh` creates in its
# namespace; and the host pushes to it from a throwaway container attached to the same network
# (see `_crane` in tests/cluster/test_admission.py), which is what keeps us out of the usual
# insecure-registry swamp -- the host's docker daemon never talks to it at all.
#
# It serves TLS, not plain HTTP (astro-mine-cloud#30). Kyverno's registry client speaks HTTPS even
# with `--allowInsecureRegistry=true` -- that flag only skips certificate *verification*, it never
# downgrades to plaintext -- so against a plain-HTTP registry it fails the TLS handshake ("server
# gave HTTP response to HTTPS client") and refuses every image for a transport reason, signed or
# not. So we mint a self-signed CA and a `kind-registry` server cert here, serve `registry:2` with
# it, and hand the CA to everyone that dials the registry (containerd on the nodes, Kyverno, crane,
# cosign). The CA never leaves this ephemeral cluster; it exists only so the signature *fetch* has
# a cert to trust. A real registry would present a publicly-trusted cert and need none of this.
#
# Port 5001 is also published on the host purely as a convenience; `docker pull localhost:5001`
# now needs the CA in the host daemon's trust to succeed (debugging only).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGISTRY_NAME="${REGISTRY_NAME:-kind-registry}"
REGISTRY_HOST_PORT="${REGISTRY_HOST_PORT:-5001}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:2}"
# Where the self-signed CA + server cert live. Shared with up.sh (the nodes' containerd trust, the
# Kyverno --set-file, the harness.env CA path) and cleaned by down.sh. Kept stable across re-runs:
# regenerating the CA under a registry the cluster already trusts would break every pull.
REGISTRY_CERT_DIR="${REGISTRY_CERT_DIR:-${HERE}/registry-certs}"

# --- 1. the registry's TLS material ----------------------------------------------------------
#
# One CA, one server cert. The SAN is `kind-registry` -- the single name the nodes, Kyverno, crane
# and cosign all dial (hostname verification strips the :5000). serverAuth EKU because Go rejects a
# cert for server use if it carries EKUs that omit it.
if [ ! -f "${REGISTRY_CERT_DIR}/ca.crt" ] || [ ! -f "${REGISTRY_CERT_DIR}/server.crt" ]; then
  mkdir -p "${REGISTRY_CERT_DIR}"
  openssl req -x509 -newkey rsa:4096 -nodes -sha256 -days 3650 \
    -keyout "${REGISTRY_CERT_DIR}/ca.key" \
    -out "${REGISTRY_CERT_DIR}/ca.crt" \
    -subj "/CN=astro-mine-kind-registry-ca"
  openssl req -newkey rsa:4096 -nodes -sha256 \
    -keyout "${REGISTRY_CERT_DIR}/server.key" \
    -out "${REGISTRY_CERT_DIR}/server.csr" \
    -subj "/CN=${REGISTRY_NAME}"
  openssl x509 -req -sha256 -days 3650 \
    -in "${REGISTRY_CERT_DIR}/server.csr" \
    -CA "${REGISTRY_CERT_DIR}/ca.crt" -CAkey "${REGISTRY_CERT_DIR}/ca.key" -CAcreateserial \
    -out "${REGISTRY_CERT_DIR}/server.crt" \
    -extfile <(printf 'subjectAltName=DNS:%s,DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\n' "${REGISTRY_NAME}")
  # World-readable: crane and cosign run as uid 65532 inside their containers and read the CA over a
  # bind mount, and the registry container reads server.{crt,key}. Throwaway keys for a cluster that
  # lives ~15 minutes; the private halves never leave the runner.
  chmod 0755 "${REGISTRY_CERT_DIR}"
  chmod 0644 "${REGISTRY_CERT_DIR}"/*.crt "${REGISTRY_CERT_DIR}"/*.key
  echo "generated a self-signed CA + ${REGISTRY_NAME} server cert in ${REGISTRY_CERT_DIR}"
fi

# --- 2. the registry container ---------------------------------------------------------------
#
# (Re)create it unless it is already running AND already serving TLS -- a registry left over from
# an older plain-HTTP harness has to be replaced, or Kyverno's HTTPS fetch fails against it exactly
# as before.
serving_tls=0
if [ "$(docker inspect -f '{{.State.Running}}' "${REGISTRY_NAME}" 2>/dev/null || true)" = 'true' ]; then
  if docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${REGISTRY_NAME}" 2>/dev/null \
      | grep -q '^REGISTRY_HTTP_TLS_CERTIFICATE='; then
    serving_tls=1
  fi
fi

if [ "${serving_tls}" != '1' ]; then
  docker rm -f "${REGISTRY_NAME}" >/dev/null 2>&1 || true
  docker run -d --restart=always \
    -p "127.0.0.1:${REGISTRY_HOST_PORT}:5000" \
    -v "${REGISTRY_CERT_DIR}:/certs:ro" \
    -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/server.crt \
    -e REGISTRY_HTTP_TLS_KEY=/certs/server.key \
    --name "${REGISTRY_NAME}" \
    "${REGISTRY_IMAGE}"
  echo "registry ${REGISTRY_NAME} started (TLS)"
else
  echo "registry ${REGISTRY_NAME} already running (TLS)"
fi
