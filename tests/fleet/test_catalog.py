"""The Fleet/Hub catalog as a selectable robot menu + capability declarations (FLEET-11).

Exercises the acceptance criteria against a temporary Hub OCI-layout registry (pytest
``tmp_path``): a Hub-published asset appears in the menu with its **vehicle kind** and the Core
**capability tags** it declares; ``requires`` filters delegate to Core's negotiation rule; a
brand-new vehicle kind surfaces with its glTF/USD **geometry preview** and **no Fleet code
change**; and a mass-model asset previews empty rather than erroring.
"""

from __future__ import annotations

import pytest

from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.sadf import SadfDocument, load_sadf
from astro_mine.core.sadf.enums import GeometryFormat, GeometryRole
from astro_mine.core.sadf.model import GeometryRef
from astro_mine.fleet._core import canonical_json
from astro_mine.fleet.capabilities import CapabilityError
from astro_mine.fleet.catalog import (
    MenuEntry,
    asset_preview,
    list_menu,
    materialize_preview,
)
from astro_mine.fleet.library import load_reference
from astro_mine.fleet.packaging.hub import HubError, publish_asset
from astro_mine.hub.registry import open_registry
from astro_mine.hub.supply_chain import generate_keypair


def _publish_roster(registry, names):
    """Sign+publish each reference asset in ``names`` to ``registry``."""
    private_pem, _ = generate_keypair()
    for name in names:
        publish_asset(load_reference(name), open_registry(str(registry)), sign_key=private_pem)


def _novel_geometry_asset() -> SadfDocument:
    """A brand-new vehicle kind (not in the shipped library) carrying glTF + USD geometry refs.

    Derived from a valid reference so it round-trips Core validation; the point is that its
    ``identity.kind`` ("hopper") is one Fleet has never seen, so its appearance in the menu proves
    the no-code-change discovery path.
    """
    cp = load_reference("relay_orbiter").model_copy(deep=True)
    cp.asset.identity.id = "hopper-mk1"
    cp.asset.identity.name = "Hopper Mk1"
    cp.asset.identity.kind = "hopper"
    cp.asset.geometry.append(
        GeometryRef(
            role=GeometryRole.VISUAL,
            format=GeometryFormat.GLTF,
            uri="hopper.glb",
            frame=cp.asset.root_frame,
        )
    )
    cp.asset.geometry.append(
        GeometryRef(
            role=GeometryRole.VISUAL,
            format=GeometryFormat.USD,
            uri="hopper.usda",
            frame=cp.asset.root_frame,
        )
    )
    return load_sadf(canonical_json(cp))


def _publish_geometry_asset(registry, base_dir) -> str:
    """Publish a signed, geometry-bearing 'hopper' asset (real glTF + USD blobs); return its ref.

    Unlike :func:`_novel_geometry_asset`, this writes actual geometry files and publishes with a
    ``base_dir`` so the glTF/USD bytes ride along as OCI layers — the precondition for a
    *renderable* preview (``materialize_preview``), not just resolvable refs.
    """
    (base_dir / "geometry").mkdir(parents=True, exist_ok=True)
    (base_dir / "geometry" / "hopper.glb").write_bytes(b"GLB-BYTES-123")
    (base_dir / "geometry" / "hopper.usda").write_bytes(b"USDA-BYTES-123")
    cp = load_reference("relay_orbiter").model_copy(deep=True)
    cp.asset.identity.id = "hopper-mk1"
    cp.asset.identity.name = "Hopper Mk1"
    cp.asset.identity.kind = "hopper"
    for fmt, uri in (
        (GeometryFormat.GLTF, "geometry/hopper.glb"),
        (GeometryFormat.USD, "geometry/hopper.usda"),
    ):
        cp.asset.geometry.append(
            GeometryRef(role=GeometryRole.VISUAL, format=fmt, uri=uri, frame=cp.asset.root_frame)
        )
    private_pem, _ = generate_keypair()
    publish_asset(load_sadf(canonical_json(cp)), open_registry(str(registry)),
        sign_key=private_pem, base_dir=base_dir)
    return "hopper-mk1:0.1.0"


def _publish_wheeled_policy(registry) -> None:
    """Publish a ``kind: policy`` artifact declaring robot capability tags.

    This is not a malformed manifest — it is Core's own shipped example
    (``examples/plugins/greedy-prospecting-baseline.manifest.yaml``), which declares
    ``prospecting.neutron`` + ``mobility.wheeled`` on a policy. Capability tags are not exclusive
    to assets, so a robot menu that trusts them alone will offer a policy as a vehicle.
    """
    from astro_mine.core.registry import PluginManifest
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import open_registry

    manifest = PluginManifest(
        name="greedy-prospecting-baseline",
        version="0.1.0",
        kind=PluginKind.POLICY,
        description="Greedy active-perception baseline (Core's shipped example policy).",
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
        capability_tags=["prospecting.neutron", "mobility.wheeled"],
    )
    private_pem, _ = generate_keypair()
    HubClient(open_registry(registry)).publish(
        name=manifest.name,
        version=manifest.version,
        kind="plugin",
        manifest=manifest,
        private_key_pem=private_pem,
    )


def test_the_menu_is_assets_only_even_when_another_kind_declares_robot_tags(tmp_path):
    """A policy carrying `mobility.wheeled` must not appear in the robot menu (#49).

    The menu is the taskability query — "who can dig" — so a non-asset in it is a category error:
    Mind and Allocate would be handed a candidate that cannot be tasked, and Studio's robot picker
    would offer a policy as a vehicle. Registry-wide discovery across kinds is `hub search`'s job.
    """
    reg = tmp_path / "reg"
    _publish_roster(reg, ["prospecting_rover"])
    _publish_wheeled_policy(reg)

    # The policy is in the registry — this is a menu filter, not a publish failure.
    from astro_mine.hub.client import catalog_from_registry
    from astro_mine.hub.registry import Registry, open_registry

    assert {e.kind for e in catalog_from_registry(Registry(reg)).all()} == {"asset", "policy"}

    menu = list_menu(open_registry(str(reg)))
    assert [m.reference for m in menu] == ["prospecting-rover:0.1.0"]
    # ...and the tag it shares with the rover does not pull it back in.
    wheeled = list_menu(open_registry(str(reg)), requires=["mobility.wheeled"])
    assert [m.reference for m in wheeled] == ["prospecting-rover:0.1.0"]


def test_menu_lists_published_assets_with_vehicle_kind_and_tags(tmp_path):
    reg = tmp_path / "reg"
    _publish_roster(reg, ["excavator", "relay_orbiter", "prospecting_rover"])

    menu = list_menu(open_registry(str(reg)))
    assert [m.reference for m in menu] == sorted(m.reference for m in menu)  # deterministic order
    assert all(isinstance(m, MenuEntry) for m in menu)

    by_kind = {m.kind: m for m in menu}
    # The menu groups by the *vehicle* kind (attributes["asset_kind"]), never the plugin kind.
    assert set(by_kind) == {"excavator", "orbiter", "rover"}
    orbiter = by_kind["orbiter"]
    assert orbiter.name == "Relay Orbiter"
    assert orbiter.namespace == "open"
    assert orbiter.digest.startswith("sha256:")
    # Capability tags are the Core-vocabulary declarations Mind/Allocate reason over.
    assert "comms.relay" in orbiter.capability_tags
    assert "mobility.orbiter" in orbiter.capability_tags


def test_requires_filters_by_declared_capability(tmp_path):
    reg = tmp_path / "reg"
    _publish_roster(reg, ["excavator", "relay_orbiter", "prospecting_rover"])

    # Single tag: only wheeled ground assets (Core's satisfies rule, not Fleet planner logic).
    assert {m.kind for m in list_menu(open_registry(str(reg)),
        requires=["mobility.wheeled"])} == {"excavator", "rover"}
    assert {m.kind for m in list_menu(open_registry(str(reg)),
        requires=["comms.relay"])} == {"orbiter", "rover"}
    # Multiple tags are ANDed: only the prospecting rover both drives and drills.
    both = list_menu(open_registry(str(reg)), requires=["mobility.wheeled", "excavation.drill"])
    assert [m.kind for m in both] == ["rover"]


def test_requires_rejects_a_tag_outside_cores_vocabulary(tmp_path):
    reg = tmp_path / "reg"
    _publish_roster(reg, ["relay_orbiter"])
    # An unknown tag is a Core RFC, never a Fleet-private extension — fail loudly.
    with pytest.raises(CapabilityError):
        list_menu(open_registry(str(reg)), requires=["not.a.real.tag"])


def test_new_hub_published_type_appears_with_preview_and_no_fleet_change(tmp_path):
    """The Fleet Phase-1 exit criterion: a new vehicle kind arrives as content, not a code edit."""
    reg = tmp_path / "reg"
    private_pem, _ = generate_keypair()
    publish_asset(_novel_geometry_asset(), open_registry(str(reg)), sign_key=private_pem)

    entry = {m.reference: m for m in list_menu(open_registry(str(reg)))}["hopper-mk1:0.1.0"]
    assert entry.kind == "hopper"  # a kind Fleet's code has never enumerated
    assert entry.name == "Hopper Mk1"

    # The single-asset preview widget (VIEW-03) resolves its geometry by format.
    gltf = asset_preview(open_registry(str(reg)), entry.reference, fmt="gltf")
    assert [(r.role.value, r.uri) for r in gltf] == [("visual", "hopper.glb")]
    usd = asset_preview(open_registry(str(reg)), entry.reference, fmt=GeometryFormat.USD)
    assert [r.uri for r in usd] == ["hopper.usda"]


def test_preview_of_a_mass_model_asset_is_empty(tmp_path):
    reg = tmp_path / "reg"
    _publish_roster(reg, ["relay_orbiter"])  # reference assets ship no geometry
    assert asset_preview(open_registry(str(reg)), "relay-orbiter:0.1.0") == []


def test_empty_registry_yields_an_empty_menu(tmp_path):
    (tmp_path / "reg").mkdir()
    assert list_menu(open_registry(str(tmp_path / "reg"))) == []


def test_materialize_preview_writes_a_servable_document_and_geometry(tmp_path):
    reg = tmp_path / "reg"
    ref = _publish_geometry_asset(reg, tmp_path / "src")

    out = tmp_path / "served"
    document = materialize_preview(open_registry(str(reg)), ref, out)

    # The returned path is the SADF-JSON documentUrl target, at the served-dir root.
    assert document.parent == out
    assert document.name == "hopper-mk1.sadf.json"
    assert load_sadf(document.read_bytes()).asset.identity.id == "hopper-mk1"
    # Geometry blobs are laid out at each ref's relative uri, byte-identical to what was published,
    # so the View widget resolves new URL(ref.uri, documentUrl) to the right bytes.
    assert (out / "geometry" / "hopper.glb").read_bytes() == b"GLB-BYTES-123"
    assert (out / "geometry" / "hopper.usda").read_bytes() == b"USDA-BYTES-123"


def test_materialize_preview_of_mass_model_writes_only_the_document(tmp_path):
    reg = tmp_path / "reg"
    _publish_roster(reg, ["relay_orbiter"])  # geometry-less reference asset

    out = tmp_path / "served"
    document = materialize_preview(open_registry(str(reg)),
        "relay-orbiter:0.1.0", out)
    assert document.is_file()
    assert list(out.rglob("*.glb")) == []  # no geometry blobs to serve


def test_materialize_preview_rejects_a_path_traversal_uri(tmp_path):
    reg = tmp_path / "reg"
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "escape.glb").write_bytes(b"PWNED")  # sits at src/../escape.glb
    cp = load_reference("relay_orbiter").model_copy(deep=True)
    cp.asset.identity.id = "evil"
    cp.asset.geometry.append(
        GeometryRef(
            role=GeometryRole.VISUAL,
            format=GeometryFormat.GLTF,
            uri="../escape.glb",
            frame=cp.asset.root_frame,
        )
    )
    private_pem, _ = generate_keypair()
    publish_asset(load_sadf(canonical_json(cp)), open_registry(str(reg)), sign_key=private_pem,
        base_dir=src)

    # Core does not reject a "../" geometry uri, so materialize_preview must fail closed itself.
    with pytest.raises(HubError, match="escapes the output directory"):
        materialize_preview(open_registry(str(reg)), "evil:0.1.0", tmp_path / "served")
