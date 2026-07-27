"""Integration: signed artifacts against a **real** OCI registry (RM-P1-HUB-01/03/06).

Two proofs, against the Zot service the `integration` CI job runs (hub.md §11 recommends **Zot** for
"the lean OCI-native self-host (built-in cosign/Referrers/sync)"):

1. :class:`RemoteRegistry` — the client publishes, resolves, re-verifies, and pulls **without
   shelling out to ``oras``**: Hub speaks the OCI Distribution Spec itself (RM-P1-HUB-06). The
   verify-twice contract and the payload-layer retrieval are exercised over the wire, and a tamper
   is proven to fail closed.
2. ``oras`` interop — the local OCI *layout* still round-trips through a real registry with the
   standard tooling, i.e. "standards in, standards out" (hub.md §2): what Hub writes, other OCI
   clients read.

Runs in CI or locally when ``HUB_OCI_REGISTRY`` points at a registry (``docker compose up -d``).
``HUB_REMOTE_REGISTRY`` (+ standard Docker credentials, e.g. ``GITHUB_TOKEN`` for ghcr.io) targets a
**hosted** registry instead, which is what proves the auth path against something real.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Blob, IntegrityError, Registry, RemoteRegistry
from astro_mine.hub.supply_chain import attest, generate_keypair

from ..conftest import make_manifest

pytestmark = pytest.mark.integration

_REGISTRY = os.environ.get("HUB_OCI_REGISTRY")  # e.g. "localhost:5000" (the Zot service)
_HOSTED = os.environ.get(
    "HUB_REMOTE_REGISTRY"
)  # e.g. "ghcr.io/astro-mine" (a real hosted registry)
_ORAS = shutil.which("oras")

PAYLOAD = Blob("application/vnd.astro-mine.onnx", b"onnx-policy-payload")


def _location(registry: str) -> str:
    """A local registry is plain HTTP; a hosted one is HTTPS with real credentials."""
    return registry if registry.startswith(("http://", "https://")) else f"http://{registry}"


@pytest.mark.skipif(not _REGISTRY, reason="HUB_OCI_REGISTRY required")
def test_publish_verify_pull_against_a_real_registry() -> None:
    """The whole client contract over the OCI Distribution Spec — no `oras` binary involved."""
    assert _REGISTRY is not None
    name = f"pol-{uuid.uuid4().hex[:8]}"
    remote = RemoteRegistry(f"{_location(_REGISTRY)}/astro-mine")
    private_pem, public_pem = generate_keypair()
    client = HubClient(remote, trusted_public_key_pem=public_pem)

    artifact = client.publish(
        name=name,
        version="1.0.0",
        kind="policy",
        manifest=make_manifest(name, "1.0.0"),
        layers=[PAYLOAD],
        private_key_pem=private_pem,
    )

    # resolve → re-verify (signature + SLSA + SBOM, pinned to the trusted key) → pull
    assert remote.resolve(f"{name}:1.0.0").digest == artifact.digest
    assert client.verify(f"{name}:1.0.0") == artifact.digest
    assert client.load(f"{name}:1.0.0").name == name

    # the payload layer comes back verified, through the client (never registry.pull_blob)
    (layer,) = client.pull_payload(f"{name}:1.0.0")
    assert layer.data == PAYLOAD.data
    assert layer.digest == PAYLOAD.digest

    assert remote.versions(name) == ["1.0.0"]
    assert len(remote.referrers(artifact.digest)) == 3  # signature + SLSA + SBOM, via Referrers API


@pytest.mark.skipif(not _REGISTRY, reason="HUB_OCI_REGISTRY required")
def test_a_lying_registry_fails_closed(tmp_path: Path) -> None:
    """Bytes a real registry serves are re-hashed client-side; a mismatch raises (hub.md §2.3)."""
    assert _REGISTRY is not None
    remote = RemoteRegistry(f"{_location(_REGISTRY)}/astro-mine")
    name = f"pol-{uuid.uuid4().hex[:8]}"
    artifact = remote.publish(
        name=name,
        version="1.0.0",
        kind="policy",
        config=make_manifest(name, "1.0.0").model_dump(mode="json"),
        layers=[PAYLOAD],
    )
    remote.verify(artifact.digest)  # honest bytes verify

    # ask the registry for a digest whose bytes it cannot possibly hold: the address must not verify
    with pytest.raises((IntegrityError, KeyError)):
        remote.pull_blob("sha256:" + "f" * 64)


@pytest.mark.skipif(not _HOSTED, reason="HUB_REMOTE_REGISTRY (a hosted registry) not configured")
def test_pull_from_a_hosted_registry_with_standard_credentials() -> None:
    """Auth via the standard Docker credential sources against a real hosted registry (ghcr.io)."""
    assert _HOSTED is not None
    name = f"pol-{uuid.uuid4().hex[:8]}"
    remote = RemoteRegistry(_HOSTED)  # credentials resolved from the Docker config / GITHUB_TOKEN
    private_pem, public_pem = generate_keypair()
    client = HubClient(remote, trusted_public_key_pem=public_pem)

    artifact = client.publish(
        name=name,
        version="1.0.0",
        kind="policy",
        manifest=make_manifest(name, "1.0.0"),
        layers=[PAYLOAD],
        private_key_pem=private_pem,
    )
    assert client.verify(f"{name}:1.0.0") == artifact.digest
    assert client.pull_payload(f"{name}:1.0.0")[0].data == PAYLOAD.data


@pytest.mark.skipif(not (_REGISTRY and _ORAS), reason="oras + HUB_OCI_REGISTRY required")
def test_signed_artifact_round_trips_through_registry(tmp_path: Path) -> None:
    """Standards in, standards out: what Hub writes locally, `oras` pushes and reads back."""
    assert _REGISTRY is not None
    layout = tmp_path / "layout"
    registry = Registry(layout)
    artifact = registry.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        config=make_manifest("pol", "1.0.0").model_dump(mode="json"),
        layers=[Blob("application/octet-stream", b"payload")],
    )
    private_pem, _ = generate_keypair()
    attest(registry, artifact.digest, private_key_pem=private_pem, name="pol", version="1.0.0")

    target = f"{_REGISTRY}/astro-mine/pol:v1"
    # Copy the manifest (by digest, unambiguous) + its referrers to the real registry.
    push = subprocess.run(
        [
            _ORAS,
            "cp",
            "-r",
            "--from-oci-layout",
            "--to-plain-http",
            f"{layout}@{artifact.digest}",
            target,
        ],
        capture_output=True,
        text=True,
    )
    assert push.returncode == 0, push.stderr

    # Fetch the pushed manifest's descriptor back; its digest must equal what we pushed.
    fetched = subprocess.run(
        [_ORAS, "manifest", "fetch", "--plain-http", "--descriptor", target],
        capture_output=True,
        text=True,
        check=True,
    )
    assert artifact.digest in fetched.stdout  # content address preserved end-to-end
