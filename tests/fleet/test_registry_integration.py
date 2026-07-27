"""Signature verifies before an asset loads — through Core's real registry (FLEET-06).

Exercises the acceptance criterion end-to-end: a packaged, signed OCI asset loads
through :class:`astro_mine.core.registry.PluginRegistry` with Fleet's verifier, while an
unsigned or tampered one is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.registry import PluginManifest, PluginRegistry, UnsignedManifest
from astro_mine.core.sadf import load_sadf
from astro_mine.fleet.packaging import oci, package_oci
from astro_mine.fleet.packaging.verifier import make_verifier
from astro_mine.seal import SignatureError, generate_keypair

from .conftest import VALID_SADF


def _load_manifest(layout: Path) -> PluginManifest:
    config, signature = oci.load_config_and_signature(layout)
    return PluginManifest.model_validate({**config, "signature": signature})


def test_signed_asset_loads_through_the_registry(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    artifact = package_oci(
        load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path, sign_key=private_pem
    )

    registry = PluginRegistry(
        require_signature=True, verifier=make_verifier(trusted_public_key_pem=public_pem)
    )
    loaded = registry.register(_load_manifest(artifact.path))
    assert loaded.name == "test.rover"


def test_unsigned_asset_is_refused(tmp_path: Path) -> None:
    artifact = package_oci(load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path)  # unsigned
    registry = PluginRegistry(require_signature=True, verifier=make_verifier())
    with pytest.raises(UnsignedManifest):
        registry.register(_load_manifest(artifact.path))


def test_inconsistent_digest_claim_is_rejected(tmp_path: Path) -> None:
    # A tampered *claim* (content-tampering is covered in test_verify_integrity.py):
    # mutate the signed digest so the ECDSA signature no longer matches provenance.digest.
    private_pem, public_pem = generate_keypair()
    artifact = package_oci(
        load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path, sign_key=private_pem
    )
    manifest = _load_manifest(artifact.path)
    assert manifest.provenance is not None
    manifest.provenance.digest = "sha256:" + "0" * 64  # signed payload no longer matches

    registry = PluginRegistry(
        require_signature=True, verifier=make_verifier(trusted_public_key_pem=public_pem)
    )
    with pytest.raises(SignatureError):
        registry.register(manifest)
