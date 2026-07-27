#!/usr/bin/env bash
# Tear the ephemeral cluster down. Safe to run when nothing is up -- which is what makes it usable
# from an always() CI step, where it must not turn a failed test run into a failed *job*.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-astro-mine}"
REGISTRY_NAME="${REGISTRY_NAME:-kind-registry}"
REGISTRY_CERT_DIR="${REGISTRY_CERT_DIR:-${HERE}/registry-certs}"
KEEP_REGISTRY="${KEEP_REGISTRY:-0}"

if command -v kind >/dev/null; then
  kind delete cluster --name "${CLUSTER_NAME}" || true
fi

if [ "${KEEP_REGISTRY}" != "1" ] && command -v docker >/dev/null; then
  docker rm -f "${REGISTRY_NAME}" >/dev/null 2>&1 || true
  # Drop the TLS material with the registry it belonged to. Kept in lockstep: the certs are
  # bind-mounted into a KEPT registry, so they only go when the container does (astro-mine-cloud#30).
  rm -rf "${REGISTRY_CERT_DIR}"
fi

rm -f "${HERE}/kubeconfig" "${HERE}/harness.env"
rm -rf "${HERE}/cosign"

echo "torn down"
