"""Generate the frozen surrogate ONNX-tier fixture for the Sim scheduler tests (RM-P1-SIM-03).

Sim imports **only** Core; it never imports ``astro_mine.surrogate``. So the served surrogate tier
its tests load — a raw-state ONNX bundle + its signed Core ``PluginManifest`` + the trusting public
key — is generated **offline** by this script (run from a surrogate-enabled env, the
``astro-mine-surrogate`` ``[serve]``/``[publish]`` extras) and checked in under
``tests/fixtures/surrogate/``. This mirrors the surrogate repo's own ``[datagen]`` pattern: a
sibling is imported only by a manual generator script, never by the package or CI.

Run:  uv run --extra serve --extra publish python scripts/gen_surrogate_fixture.py
      (from a checkout of astro-mine-surrogate, with this script copied in — or point PYTHONPATH
       at an installed astro-mine-surrogate). The bytes are build-specific (torch CPU is not
       bit-portable) but frozen once checked in, exactly like the surrogate's DEM .npz fixture.
"""

from __future__ import annotations

from pathlib import Path

from astro_mine.hub.supply_chain import generate_keypair, sign_digest
from astro_mine.surrogate.enums import ServedBackend
from astro_mine.surrogate.manifest import build_surrogate_manifest
from astro_mine.surrogate.models.excavation import build_excavation_surrogate
from astro_mine.surrogate.models.train import TrainConfig
from astro_mine.surrogate.serve import export_excavation_surrogate

_OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "surrogate"
_NAME = "excavation-gns"
_VERSION = "0.1.0"


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    surrogate = build_excavation_surrogate(
        config=TrainConfig(hidden=16, ensemble_size=2, epochs=40), seed=0
    )
    bundle = export_excavation_surrogate(surrogate)
    bundle_bytes = bundle.serialize()

    manifest = build_surrogate_manifest(
        name=_NAME,
        version=_VERSION,
        report=bundle.error_report,
        artifact_digest=bundle.content_hash(),
        served_backend=ServedBackend.ONNX,
    )
    private_pem, public_pem = generate_keypair()
    signed = manifest.model_copy(
        update={"signature": sign_digest(manifest.provenance.digest, private_pem)}
    )

    (_OUT / "excavation_surrogate.onnxbundle").write_bytes(bundle_bytes)
    (_OUT / "manifest.json").write_text(signed.model_dump_json(indent=2))
    (_OUT / "signer_public_key.pem").write_bytes(public_pem)
    print(f"wrote fixture to {_OUT}")
    print(f"  bundle: {len(bundle_bytes)} bytes, digest {bundle.content_hash()}")
    print(f"  budget: {bundle.error_report.substitution_policy.recommended_error_budget}")
    print(f"  trust region: {dict(bundle.error_report.trust_region.bounds)}")


if __name__ == "__main__":
    main()
