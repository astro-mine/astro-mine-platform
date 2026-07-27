"""Tests for ``astro_mine.core.registry`` — the plugin manifest contract, the document
loader + dual-use gate, and the :class:`PluginRegistry` load-time gates: version
negotiation, the signature gate, and the reused capability-tag export-control gate
(RM-P0-CORE-05)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from astro_mine.core.registry import (
    IncompatibleManifest,
    ManifestDocument,
    ManifestValidationError,
    PluginManifest,
    PluginRegistry,
    RegistryError,
    UnsignedManifest,
    load_manifest,
    load_schema,
    validate_manifest,
)
from astro_mine.core.registry.enums import (
    CapabilityTag,
    PluginKind,
    SignatureKind,
    SignatureScheme,
)
from astro_mine.core.registry.model import (
    CatalogRecord,
    CoreInterfaceVersion,
    Provenance,
    Signature,
)

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "plugins").glob("*.yaml"))


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "name": "lunar-engine",
        "version": "0.1.0",
        "kind": "regime_engine",
        "core_interfaces": {"env": "0.1.0", "sadf": "0.1.0"},
        "signature": {"scheme": "sigstore_cosign", "value": "sig"},
    }
    manifest.update(overrides)
    return manifest


def _doc(**overrides: Any) -> dict[str, Any]:
    return {"manifest_version": "0.1", "manifest": _manifest(**overrides)}


# --- the document contract --------------------------------------------------------


def test_load_manifest_round_trip() -> None:
    doc = load_manifest(yaml_dump(_doc()))
    assert isinstance(doc, ManifestDocument)
    assert doc.manifest.name == "lunar-engine"
    assert doc.manifest.kind == PluginKind.REGIME_ENGINE
    assert doc.manifest.core_interfaces == {"env": "0.1.0", "sadf": "0.1.0"}


def test_comms_model_kind_loads() -> None:
    # Link registers as a Core comms-environment plugin (C6), parallel to world_provider.
    doc = load_manifest(yaml_dump(_doc(kind="comms_model", core_interfaces={"messages": "0.1.0"})))
    assert doc.manifest.kind == PluginKind.COMMS_MODEL


@pytest.mark.parametrize("kind", ["design", "campaign"])
def test_studio_design_and_campaign_kinds_load(kind: str) -> None:
    """RFC-0008: Studio's frozen artifacts are describable by a Core manifest.

    They implement no Core interface -- `core_interfaces` is empty -- because nothing loads them as
    code. They exist so Hub indexes a published design by the Core manifest rather than a
    Studio-private schema (hub.md 2 principle 2).
    """
    doc = load_manifest(yaml_dump(_doc(kind=kind, core_interfaces={})))
    assert doc.manifest.kind == PluginKind(kind)
    assert doc.manifest.core_interfaces == {}


def test_manifest_defaults_are_empty() -> None:
    manifest = PluginManifest(name="p", version="0.1.0", kind=PluginKind.METRIC)
    assert manifest.core_interfaces == {}
    assert manifest.inputs == [] and manifest.outputs == []
    assert manifest.capability_tags == [] and manifest.regimes == []
    assert manifest.attributes == {}
    assert manifest.provenance is None and manifest.signature is None


def test_empty_document_fails_loudly() -> None:
    with pytest.raises(ManifestValidationError):
        load_manifest("{}")


def test_load_manifest_accepts_bytes() -> None:
    doc = load_manifest(yaml_dump(_doc()).encode("utf-8"))
    assert doc.manifest.version == "0.1.0"


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ManifestValidationError, match="schema validation"):
        load_manifest(yaml_dump(_doc(bogus=1)))


def test_bad_kind_is_rejected() -> None:
    with pytest.raises(ManifestValidationError, match="schema validation"):
        load_manifest(yaml_dump(_doc(kind="warp_drive")))


def test_non_mapping_is_rejected() -> None:
    with pytest.raises(ManifestValidationError, match="must be a YAML/JSON mapping"):
        load_manifest("[1, 2, 3]")


def test_bad_version_const_is_rejected() -> None:
    with pytest.raises(ManifestValidationError):
        load_manifest(yaml_dump({"manifest_version": "0.2", "manifest": _manifest()}))


# --- semantic gates (loader) ------------------------------------------------------


def test_gated_capability_tag_is_refused() -> None:
    doc = _doc(capability_tags=["mobility.wheeled", "operational_targeting"])
    with pytest.raises(ManifestValidationError, match="reserved/gated capability tag"):
        load_manifest(yaml_dump(doc))


def test_ground_truth_access_tag_is_refused() -> None:
    doc = _doc(capability_tags=["ground_truth_access"])
    with pytest.raises(ManifestValidationError, match="reserved/gated capability tag"):
        load_manifest(yaml_dump(doc))


def test_malformed_core_interface_version_is_refused() -> None:
    with pytest.raises(ManifestValidationError, match=r"MAJOR\.MINOR\.PATCH"):
        load_manifest(yaml_dump(_doc(core_interfaces={"env": "oops"})))


# --- validate_manifest entry points -----------------------------------------------


def test_validate_manifest_accepts_typed_document() -> None:
    doc = ManifestDocument.model_validate(_doc())
    validate_manifest(doc)  # no raise


def test_validate_manifest_accepts_text_and_dict() -> None:
    validate_manifest(yaml_dump(_doc()))
    validate_manifest(_doc())


def test_validate_manifest_rejects_typed_document_with_gated_tag() -> None:
    doc = ManifestDocument.model_validate(_doc(capability_tags=["operational_targeting"]))
    with pytest.raises(ManifestValidationError, match="reserved/gated"):
        validate_manifest(doc)


def test_validate_manifest_rejects_unknown_type() -> None:
    with pytest.raises(ManifestValidationError, match="cannot validate object of type"):
        validate_manifest(object())  # type: ignore[arg-type]


def test_load_schema_is_a_mapping() -> None:
    schema = load_schema()
    assert "PluginManifest" in schema["$defs"]


# --- the registry: version negotiation --------------------------------------------


def test_register_and_resolve() -> None:
    reg = PluginRegistry()
    manifest = reg.load(yaml_dump(_doc()))
    assert manifest.name == "lunar-engine"
    assert reg.resolve("lunar-engine") is manifest
    assert "lunar-engine" in reg
    assert len(reg) == 1
    assert reg.manifests == (manifest,)
    assert reg.by_kind(PluginKind.REGIME_ENGINE) == [manifest]
    assert reg.by_kind(PluginKind.METRIC) == []


def test_resolve_unknown_raises() -> None:
    with pytest.raises(RegistryError, match="no plugin named"):
        PluginRegistry().resolve("ghost")


def test_duplicate_registration_is_refused() -> None:
    reg = PluginRegistry()
    reg.load(yaml_dump(_doc()))
    with pytest.raises(RegistryError, match="already registered"):
        reg.load(yaml_dump(_doc()))


def test_unsupported_major_is_refused_clearly() -> None:
    reg = PluginRegistry()
    with pytest.raises(IncompatibleManifest, match=r"env: consumer requires 1\.0\.0"):
        reg.load(yaml_dump(_doc(core_interfaces={"env": "1.0.0"})))


def test_mismatched_pre_release_minor_is_refused() -> None:
    reg = PluginRegistry()
    with pytest.raises(IncompatibleManifest, match=r"Core provides 0\.1\.0"):
        reg.load(yaml_dump(_doc(core_interfaces={"env": "0.2.0"})))


def test_unknown_interface_is_refused() -> None:
    reg = PluginRegistry()
    with pytest.raises(IncompatibleManifest, match="unknown Core interface"):
        reg.load(yaml_dump(_doc(core_interfaces={"warpdrive": "0.1.0"})))


def test_negotiation_honors_provided_override() -> None:
    reg = PluginRegistry(provided={"env": "1.2.0"})
    manifest = reg.load(yaml_dump(_doc(core_interfaces={"env": "1.1.0"})))
    assert manifest.core_interfaces == {"env": "1.1.0"}


# --- the registry: signature gate -------------------------------------------------


def test_unsigned_manifest_is_refused_by_default() -> None:
    reg = PluginRegistry()
    with pytest.raises(UnsignedManifest, match="requires a signed manifest"):
        reg.load(yaml_dump(_doc(signature=None)))


def test_explicit_unsigned_scheme_is_refused() -> None:
    reg = PluginRegistry()
    with pytest.raises(UnsignedManifest):
        reg.load(yaml_dump(_doc(signature={"scheme": "unsigned"})))


def test_signature_can_be_waived_for_dev() -> None:
    reg = PluginRegistry(require_signature=False)
    manifest = reg.load(yaml_dump(_doc(signature=None)))
    assert manifest.signature is None


def test_verifier_runs_on_a_signed_manifest() -> None:
    seen: list[str] = []
    reg = PluginRegistry(verifier=lambda m: seen.append(m.name))
    reg.load(yaml_dump(_doc()))
    assert seen == ["lunar-engine"]


def test_verifier_can_reject() -> None:
    def reject(_: PluginManifest) -> None:
        raise RegistryError("bad signature")

    reg = PluginRegistry(verifier=reject)
    with pytest.raises(RegistryError, match="bad signature"):
        reg.load(yaml_dump(_doc()))


# --- the registry: register typed manifests + the capability gate -----------------


def test_register_accepts_a_typed_manifest() -> None:
    reg = PluginRegistry(require_signature=False)
    manifest = PluginManifest(
        name="grid-backend",
        version="0.1.0",
        kind=PluginKind.RESOURCE_FIELD_BACKEND,
        core_interfaces={"sadf": "0.1.0"},
    )
    assert reg.register(manifest) is manifest
    assert reg.resolve("grid-backend").kind == PluginKind.RESOURCE_FIELD_BACKEND


def test_register_rejects_a_code_built_gated_manifest() -> None:
    # A manifest built in code (bypassing the text loader) is still gated on register.
    reg = PluginRegistry(require_signature=False)
    from astro_mine.core.sadf.enums import CapabilityTag

    manifest = PluginManifest(
        name="targeting",
        version="0.1.0",
        kind=PluginKind.POLICY,
        capability_tags=[CapabilityTag.OPERATIONAL_TARGETING],
    )
    with pytest.raises(ManifestValidationError, match="reserved/gated"):
        reg.register(manifest)


def test_register_accepts_a_manifest_document() -> None:
    reg = PluginRegistry(require_signature=False)
    doc = ManifestDocument(
        manifest_version="0.1",
        manifest=PluginManifest(name="m", version="0.1.0", kind=PluginKind.METRIC),
    )
    assert reg.register(doc).name == "m"


# --- model surface ----------------------------------------------------------------


def test_provenance_and_signature_models() -> None:
    prov = Provenance(input_hashes=["sha256:00"], digest="sha256:11", builder_version="conv-1")
    assert prov.source_content_hashes == {}
    sig = Signature(scheme=SignatureScheme.SIGSTORE_COSIGN, value="x")
    assert sig.payload is None and sig.certificate is None


# --- the shipped examples ---------------------------------------------------------


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_manifests_load_and_register(path: Path) -> None:
    reg = PluginRegistry()
    manifest = reg.load(path.read_text(encoding="utf-8"))
    assert manifest.name == path.name.removesuffix(".manifest.yaml")


# --- Hub-indexing manifest fields (RM-P1-CORE-02) ---------------------------------


def test_license_and_plural_signatures_load() -> None:
    doc = load_manifest(
        yaml_dump(
            _doc(
                license="Apache-2.0",
                signatures=[
                    {"scheme": "sigstore_cosign", "kind": "slsa_provenance", "digest": "sha256:dd"},
                    {"scheme": "sigstore_cosign", "kind": "sbom", "digest": "sha256:ee"},
                ],
            )
        )
    )
    m = doc.manifest
    assert m.license == "Apache-2.0"
    assert [s.kind for s in m.signatures] == [SignatureKind.SLSA_PROVENANCE, SignatureKind.SBOM]
    assert m.signatures[1].digest == "sha256:ee"


def test_signature_kind_defaults_to_cosign() -> None:
    sig = Signature(scheme=SignatureScheme.SIGSTORE_COSIGN, value="x")
    assert sig.kind is SignatureKind.COSIGN_SIGNATURE
    assert sig.digest is None


def test_new_manifest_fields_default_empty() -> None:
    m = PluginManifest(name="p", version="0.1.0", kind=PluginKind.METRIC)
    assert m.license is None
    assert m.signatures == []
    assert m.all_signatures == []


def test_all_signatures_folds_singular_then_plural() -> None:
    m = PluginManifest(
        name="p",
        version="0.1.0",
        kind=PluginKind.POLICY,
        signature=Signature(scheme=SignatureScheme.SIGSTORE_COSIGN),
        signatures=[
            Signature(scheme=SignatureScheme.SIGSTORE_COSIGN, kind=SignatureKind.SLSA_PROVENANCE),
            Signature(scheme=SignatureScheme.SIGSTORE_COSIGN, kind=SignatureKind.SBOM),
        ],
    )
    assert [s.kind for s in m.all_signatures] == [
        SignatureKind.COSIGN_SIGNATURE,
        SignatureKind.SLSA_PROVENANCE,
        SignatureKind.SBOM,
    ]


def test_satisfies_negotiates_interfaces_and_capabilities() -> None:
    m = PluginManifest(
        name="p",
        version="0.1.0",
        kind=PluginKind.POLICY,
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
        capability_tags=[CapabilityTag.EXCAVATION_DRILL, CapabilityTag.MOBILITY_WHEELED],
    )
    assert m.satisfies(
        interfaces={"policy": "0.1.0"}, capability_tags=[CapabilityTag.EXCAVATION_DRILL]
    )
    assert m.satisfies()  # empty constraint set is trivially satisfied
    assert not m.satisfies(interfaces={"sadf": "0.1.0"})  # interface not declared
    assert not m.satisfies(interfaces={"policy": "0.2.0"})  # pre-1.0 minor must match exactly
    assert not m.satisfies(capability_tags=["prospecting.neutron"])  # tag not declared


def test_to_catalog_record_projects_purely_from_the_manifest() -> None:
    m = load_manifest(
        yaml_dump(
            _doc(
                license="Apache-2.0",
                inputs=["Observation"],
                outputs=["ActionBatch"],
                capability_tags=["prospecting.neutron", "mobility.wheeled"],
                signatures=[{"scheme": "sigstore_cosign", "kind": "sbom", "digest": "sha256:ee"}],
            )
        )
    ).manifest
    record = m.to_catalog_record()
    assert isinstance(record, CatalogRecord)
    assert record.kind is PluginKind.REGIME_ENGINE
    assert record.license == "Apache-2.0"
    # core_interfaces{dict} projects to the sorted core_interface_versions[] array Hub indexes
    assert record.core_interface_versions == [
        CoreInterfaceVersion(interface="env", version="0.1.0"),
        CoreInterfaceVersion(interface="sadf", version="0.1.0"),
    ]
    # both the legacy singular signature and the plural set fold into signatures[]
    assert [s.kind for s in record.signatures] == [
        SignatureKind.COSIGN_SIGNATURE,
        SignatureKind.SBOM,
    ]
    # the record carries only Core-manifest-derived fields — no Hub-private facets
    assert set(CatalogRecord.model_fields) == {
        "name",
        "version",
        "kind",
        "core_interface_versions",
        "capability_tags",
        "inputs",
        "outputs",
        "license",
        "provenance",
        "signatures",
    }


def yaml_dump(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data)
