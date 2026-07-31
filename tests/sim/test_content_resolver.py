"""Content-pinned Scenario construction — the RM-P1-SIM-01 resolver.

Exercises the resolver end-to-end against a real ``astro-mine-hub`` local OCI-layout registry:
publish signed Worlds/Fleet/Prospect fixtures, resolve them **by content hash**, materialize a
fleet asset into an ``AgentSpec``, reconstruct world/resource providers via injected factories, and
prove the run reproduces (identical content hashes) and carries them in provenance.

Also asserts the narrow waist (conventions.md §1.1; bench.md §2.2): ``astro_mine.sim`` imports no
**producer** package (Worlds/Fleet/Prospect) anywhere; only the injected ``sim/bench/`` runner
adapter may import **Bench** (the direction is one-way — Bench never imports Sim); and Sim's own
runtime never imports that adapter, so the base wheel stays Bench-free.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Mapping
from typing import Any

import pytest

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.sadf.model import Asset, Identity, PowerBudget, SadfDocument
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Blob, IntegrityError, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair
from astro_mine.sim.runtime import Scenario, agent_spec_from_asset, run_episode
from astro_mine.sim.runtime import content as content_module
from astro_mine.sim.runtime._hub_adapter import HubBundleStore, open_bundle_store
from astro_mine.sim.runtime.content import (
    ContentPin,
    ContentResolver,
    ScenarioContent,
    describe_unresolved,
)

_SADF_JSON = "application/vnd.astro-mine.sadf+json"
_WORLD_TAR = "application/vnd.astro-mine.world.bundle.v1.tar"
_FIELD_TAR = "application/vnd.astro-mine.resource-field.bundle.v1.tar"


def _asset_document() -> SadfDocument:
    """A minimal but real Core SADF document — a rover with a power budget to prove the
    fleet-sourced fields flow through :func:`agent_spec_from_asset`."""
    asset = Asset(
        identity=Identity(
            id="astro-mine.fleet.test-rover", name="Test Rover", version="0.1.0", kind="rover"
        ),
        root_frame="body",
        power=PowerBudget(floor_w=5.0),
    )
    return SadfDocument(sadf_version="0.1", asset=asset)


class _FakeWorld:
    """Stand-in for a producer's ``WorldProvider`` — the resolver treats it opaquely."""

    def __init__(self, layers: Mapping[str, bytes]) -> None:
        self.layers = dict(layers)


class _FakeField:
    def __init__(self, layers: Mapping[str, bytes]) -> None:
        self.layers = dict(layers)


def _world_factory(manifest: PluginManifest, layers: Mapping[str, bytes]) -> _FakeWorld:
    return _FakeWorld(layers)


def _field_factory(manifest: PluginManifest, layers: Mapping[str, bytes]) -> _FakeField:
    return _FakeField(layers)


def _publish(
    client: HubClient,
    key: bytes,
    *,
    name: str,
    artifact_kind: str,
    manifest_kind: PluginKind,
    core_interface: str,
    layers: list[Blob],
) -> str:
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        kind=manifest_kind,
        core_interfaces={core_interface: "0.1.0"},
        license="Apache-2.0",
    )
    artifact = client.publish(
        name=name,
        version="0.1.0",
        kind=artifact_kind,
        manifest=manifest,
        layers=layers,
        private_key_pem=key,
    )
    return artifact.digest


@pytest.fixture
def published(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Publish a fleet asset + a world + a resource field to a temp registry; return the store,
    the digests, and the injected provider factories."""
    private_pem, public_pem = generate_keypair()
    registry = Registry(tmp_path / "reg")
    client = HubClient(registry, trusted_public_key_pem=public_pem)

    sadf_layer = Blob(_SADF_JSON, _asset_document().model_dump_json().encode("utf-8"))
    fleet_digest = _publish(
        client,
        private_pem,
        name="astro-mine.fleet.test-rover",
        artifact_kind="asset",
        manifest_kind=PluginKind.ASSET,
        core_interface="sadf",
        layers=[sadf_layer],
    )
    world_digest = _publish(
        client,
        private_pem,
        name="test-world",
        artifact_kind="world",
        manifest_kind=PluginKind.WORLD_PROVIDER,
        core_interface="world_provider",
        layers=[Blob(_WORLD_TAR, b"world-bundle-tar-bytes")],
    )
    field_digest = _publish(
        client,
        private_pem,
        name="test-field",
        artifact_kind="plugin",
        manifest_kind=PluginKind.RESOURCE_FIELD_BACKEND,
        core_interface="resource_field",
        layers=[Blob(_FIELD_TAR, b"field-bundle-tar-bytes")],
    )
    return {
        "store": HubBundleStore(client),
        "registry": registry,
        "fleet_digest": fleet_digest,
        # The *layer* digest (not the artifact's): the content address of the SADF payload blob, and
        # so the on-disk blob a tamper test overwrites.
        "fleet_layer_digest": sadf_layer.digest,
        "world_digest": world_digest,
        "field_digest": field_digest,
        "factories": {
            PluginKind.WORLD_PROVIDER.value: _world_factory,
            PluginKind.RESOURCE_FIELD_BACKEND.value: _field_factory,
        },
    }


def _content(published: dict[str, Any]) -> ScenarioContent:
    return ScenarioContent(
        world=ContentPin(id="test-world", reference=published["world_digest"]),
        fleet=(ContentPin(id="astro-mine.fleet.test-rover", reference=published["fleet_digest"]),),
        prospect=(ContentPin(id="test-field", reference=published["field_digest"]),),
    )


def test_resolves_fleet_asset_and_providers(published: dict[str, Any]) -> None:
    resolver = ContentResolver(published["store"], provider_factories=published["factories"])
    resolved = resolver.resolve(_content(published))

    # Fleet asset materialized from the SADF JSON layer, pinned by its digest.
    asset = resolved.assets["astro-mine.fleet.test-rover"]
    assert asset.asset.identity.id == "astro-mine.fleet.test-rover"
    assert asset.content_hash == published["fleet_digest"]

    # World + resource providers reconstructed via the injected factories, fed the pulled layers.
    assert isinstance(resolved.world_provider, _FakeWorld)
    assert resolved.world_provider.layers[_WORLD_TAR] == b"world-bundle-tar-bytes"
    assert isinstance(resolved.resource_field, _FakeField)
    assert resolved.resource_field.layers[_FIELD_TAR] == b"field-bundle-tar-bytes"

    # Every pin rides in the content-hash map.
    assert resolved.content_hashes["astro-mine.fleet.test-rover"] == published["fleet_digest"]
    assert resolved.content_hashes["test-world"] == published["world_digest"]
    assert resolved.content_hashes["test-field"] == published["field_digest"]


def test_missing_factory_leaves_provider_unresolved(published: dict[str, Any]) -> None:
    # No factories registered: the reference still resolves (hash rides in provenance) but the live
    # provider is left None for the caller to inject.
    resolver = ContentResolver(published["store"], provider_factories={})
    resolved = resolver.resolve(_content(published))
    assert resolved.world_provider is None
    assert resolved.resource_field is None
    assert resolved.content_hashes["test-world"] == published["world_digest"]


def test_agent_spec_sources_fleet_fields(published: dict[str, Any]) -> None:
    resolver = ContentResolver(published["store"], provider_factories=published["factories"])
    resolved = resolver.resolve(_content(published))
    spec = agent_spec_from_asset(
        resolved.assets["astro-mine.fleet.test-rover"],
        agent_id="rover",
        initial_position_m=(1.0, 2.0, 3.0),
        battery_soc_j=1000.0,
    )
    assert spec.agent_id == "rover"
    assert spec.initial_position_m == (1.0, 2.0, 3.0)
    # The power budget came from the resolved SADF asset, not an inline literal.
    assert spec.power is not None and spec.power.floor_w == 5.0


def test_resolution_is_deterministic(published: dict[str, Any]) -> None:
    resolver = ContentResolver(published["store"], provider_factories=published["factories"])
    first = resolver.resolve(_content(published))
    second = resolver.resolve(_content(published))
    assert first.content_hashes == second.content_hashes


def test_run_provenance_carries_content_hashes(published: dict[str, Any]) -> None:
    resolver = ContentResolver(published["store"], provider_factories=published["factories"])
    resolved = resolver.resolve(_content(published))
    spec = agent_spec_from_asset(resolved.assets["astro-mine.fleet.test-rover"], agent_id="rover")
    scenario = Scenario(name="content-pinned", agents=(spec,), horizon_steps=2)

    trace = run_episode(scenario, content_hashes=resolved.content_hashes)
    hashes = trace.provenance["source_content_hashes"]
    assert hashes["astro-mine.fleet.test-rover"] == published["fleet_digest"]
    assert hashes["test-world"] == published["world_digest"]
    # The open-loop run without content hashes is byte-identical to before this feature.
    baseline = run_episode(scenario)
    assert "test-world" not in baseline.provenance["source_content_hashes"]


def test_duplicate_fleet_ids_rejected(published: dict[str, Any]) -> None:
    pin = ContentPin(id="dup", reference=published["fleet_digest"])
    with pytest.raises(ValueError, match="duplicate fleet content ids"):
        ScenarioContent(fleet=(pin, pin))


def _tamper_fleet_layer(published: dict[str, Any]) -> None:
    """Corrupt the fleet asset's payload blob **in the registry**: the SADF bytes on disk no longer
    hash to the digest the signed manifest committed to.

    The exact tamper Hub's own ``test_payload_retrieval_fails_closed_on_a_tampered_layer`` stages —
    an attacker with write access to the OCI layout (or a bit-rotted blob), which leaves the signed
    manifest and its referrers untouched, so *only* a content-address re-check catches it."""
    registry: Registry = published["registry"]
    layer = blob_path(pathlib.Path(registry.path), published["fleet_layer_digest"])
    layer.write_bytes(b"malicious")


def test_pull_layers_round_trips_then_fails_closed_on_a_tampered_layer(
    published: dict[str, Any],
) -> None:
    # Happy path first: an untampered store hands back the published payload verbatim.
    store: HubBundleStore = published["store"]
    reference: str = published["fleet_digest"]
    assert SadfDocument.model_validate_json(store.pull_layers(reference)[_SADF_JSON]).asset

    # Tampered: the layer's bytes no longer match the digest the verified manifest commits to, so
    # retrieval fails closed (conventions.md §9) rather than returning them. Either exception is
    # correct — which one fires depends on whether the supply-chain check or the per-layer
    # content-address check reaches the bad bytes first.
    _tamper_fleet_layer(published)
    with pytest.raises((IntegrityError, SupplyChainError)):
        store.pull_layers(reference)


def test_pull_layers_enforces_content_addresses_even_unverified(published: dict[str, Any]) -> None:
    # The sharp edge, and the regression this guards: ``verify=False`` relaxes only the
    # *supply-chain* re-check (signature/SLSA/SBOM) — each layer's own content address is still
    # enforced, because the bytes come off the client's verified payload path rather than a raw
    # ``registry.pull_blob()`` (which, on the local OCI layout, is a plain ``read_bytes()`` and
    # would have handed the tampered SADF straight to :func:`_decode_asset`).
    store: HubBundleStore = published["store"]
    _tamper_fleet_layer(published)
    with pytest.raises(IntegrityError):
        store.pull_layers(published["fleet_digest"], verify=False)

    # And the flag now actually *reaches* the layer pull: an unverified resolver still fails closed.
    unverified = ContentResolver(store, provider_factories=published["factories"], verify=False)
    with pytest.raises(IntegrityError):
        unverified.resolve(_content(published))


def test_resolver_fails_closed_over_a_tampered_store(published: dict[str, Any]) -> None:
    # End to end: no corrupted asset is ever materialized — the resolver raises before a tampered
    # SADF layer can be decoded into a Core Asset (and thus into an AgentSpec / a run's provenance).
    resolver = ContentResolver(published["store"], provider_factories=published["factories"])
    _tamper_fleet_layer(published)
    with pytest.raises((IntegrityError, SupplyChainError)):
        resolver.resolve(_content(published))


def test_link_pin_rides_in_content_hashes(published: dict[str, Any]) -> None:
    resolver = ContentResolver(published["store"], provider_factories=published["factories"])
    content = ScenarioContent(
        fleet=(ContentPin(id="astro-mine.fleet.test-rover", reference=published["fleet_digest"]),),
        link=ContentPin(id="test-link", reference=published["field_digest"]),
    )
    resolved = resolver.resolve(content)
    assert resolved.content_hashes["test-link"] == published["field_digest"]


def test_fleet_bundle_without_sadf_layer_errors(tmp_path: pathlib.Path) -> None:
    private_pem, public_pem = generate_keypair()
    client = HubClient(Registry(tmp_path / "reg"), trusted_public_key_pem=public_pem)
    digest = _publish(
        client,
        private_pem,
        name="broken-asset",
        artifact_kind="asset",
        manifest_kind=PluginKind.ASSET,
        core_interface="sadf",
        layers=[Blob("application/octet-stream", b"not-a-sadf-document")],
    )
    resolver = ContentResolver(HubBundleStore(client), provider_factories={})
    content = ScenarioContent(fleet=(ContentPin(id="broken", reference=digest),))
    with pytest.raises(ValueError, match="no SADF document layer"):
        resolver.resolve(content)


def test_default_factory_discovery_via_entry_points(
    published: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # With no explicit provider_factories, the resolver discovers producer factories from the
    # ``astro_mine.providers`` entry-point group; simulate a registered Worlds producer.
    class _EntryPoint:
        name = PluginKind.WORLD_PROVIDER.value

        @staticmethod
        def load() -> Any:
            return _world_factory

    monkeypatch.setattr(
        content_module,
        "entry_points",
        lambda group: [_EntryPoint()] if group == content_module.PROVIDER_ENTRY_POINT_GROUP else [],
    )
    resolver = ContentResolver(published["store"])  # no provider_factories -> entry-point discovery
    resolved = resolver.resolve(_content(published))
    assert isinstance(resolved.world_provider, _FakeWorld)


def test_open_bundle_store_round_trip(tmp_path: pathlib.Path) -> None:
    private_pem, public_pem = generate_keypair()
    reg_path = tmp_path / "reg"
    client = HubClient(Registry(reg_path), trusted_public_key_pem=public_pem)
    sadf = _asset_document().model_dump_json().encode("utf-8")
    digest = _publish(
        client,
        private_pem,
        name="astro-mine.fleet.test-rover",
        artifact_kind="asset",
        manifest_kind=PluginKind.ASSET,
        core_interface="sadf",
        layers=[Blob(_SADF_JSON, sadf)],
    )
    store = open_bundle_store(reg_path, trusted_public_key_pem=public_pem)
    resolver = ContentResolver(store, provider_factories={})
    resolved = resolver.resolve(
        ScenarioContent(fleet=(ContentPin(id="astro-mine.fleet.test-rover", reference=digest),))
    )
    assert resolved.assets["astro-mine.fleet.test-rover"].asset.identity.id == (
        "astro-mine.fleet.test-rover"
    )


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Every module a source file imports, parsed from the AST.

    Via the AST, not a text grep, so a docstring that *names* a forbidden package to explain the
    boundary does not trip the check — only real import statements count."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def _imports_any(modules: list[str], packages: tuple[str, ...]) -> bool:
    return any(m == pkg or m.startswith(f"{pkg}.") for m in modules for pkg in packages)


def test_sim_imports_no_producer_package_anywhere() -> None:
    # The narrow waist (conventions.md §1.1): Sim consumes Worlds/Fleet/Prospect content as **Core**
    # artifacts — a SADF document, a WorldProvider, a ResourceField — reconstructed by their
    # producers' registered entry-point factories. It never imports the producers themselves, so a
    # Sim install needs none of them present. This holds for *every* file in the package.
    import astro_mine.sim

    root = pathlib.Path(astro_mine.sim.__file__).resolve().parent
    producers = ("astro_mine.worlds", "astro_mine.fleet", "astro_mine.prospect")
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if _imports_any(_imported_modules(path), producers)
    ]
    assert offenders == [], f"sim source imports a producer package: {offenders}"


def test_only_the_injected_bench_runner_imports_bench() -> None:
    # Bench is a *consumer*, not a producer, and the dependency direction is one-way by design
    # (bench.md §2.2): Bench never imports Sim, so the runner satisfying Bench's Core-typed
    # EpisodeRunner/Runner seams lives HERE and is injected into Bench. That adapter — and only that
    # adapter — may import Bench; it is optional (`astro-mine-sim[bench]`) and off the runtime path.
    import astro_mine.sim

    root = pathlib.Path(astro_mine.sim.__file__).resolve().parent
    adapter = root / "bench"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if adapter not in path.parents
        and _imports_any(_imported_modules(path), ("astro_mine.bench",))
    ]
    assert offenders == [], f"only astro_mine/sim/bench/ may import Bench; found: {offenders}"


def test_sims_runtime_never_imports_the_bench_adapter() -> None:
    # The other half of that boundary: because the adapter imports Bench, nothing in Sim's own
    # runtime may import the *adapter* — otherwise the base wheel would drag Bench in and the
    # `[bench]` extra would be a fiction. A caller reaches it explicitly; Sim never does.
    #
    # The CLI entry `__main__.py` is exempt: its `run` subcommand *is* such an explicit caller (it
    # lazily imports the adapter inside the handler), and it is off the `import astro_mine.sim` path
    # — nothing imports `__main__`, so the base wheel stays Bench-free. Verified: `import
    # astro_mine.sim` pulls in neither `__main__` nor `astro_mine.bench`.
    import astro_mine.sim

    root = pathlib.Path(astro_mine.sim.__file__).resolve().parent
    adapter = root / "bench"
    cli_entry = root / "__main__.py"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if adapter not in path.parents
        and path != cli_entry
        and _imports_any(_imported_modules(path), ("astro_mine.sim.bench",))
    ]
    assert offenders == [], f"sim runtime imports the Bench adapter: {offenders}"


# --- a blind run says so (#67) --------------------------------------------------------------


def test_an_unresolved_provider_is_recorded_not_swallowed(published: dict[str, Any]) -> None:
    """The pin resolved by digest and rebuilt nothing — and the resolver now says which, and why.

    Leaving the provider `None` is a deliberate seam (a caller may inject its own), so the fix is
    not to raise here. What was missing is the ability to tell "the caller will inject" apart from
    "nobody will, and this run is blind" (#67).
    """
    resolved = ContentResolver(published["store"], provider_factories={}).resolve(
        _content(published)
    )

    assert resolved.world_provider is None  # the seam is unchanged
    kinds = {u.kind for u in resolved.unresolved}
    assert kinds == {"world_provider", "resource_field_backend"}
    world = next(u for u in resolved.unresolved if u.kind == "world_provider")
    assert world.producer == "astro-mine-worlds"  # the package that supplies the entry point
    assert world.content_id == "test-world"
    assert "nights_survived" in world.consequence  # names the metric that stops scoring


def test_a_fully_resolved_run_records_nothing_unresolved(published: dict[str, Any]) -> None:
    resolved = ContentResolver(
        published["store"], provider_factories=published["factories"]
    ).resolve(_content(published))
    assert resolved.unresolved == ()


def test_fleet_only_content_never_reports_unresolved(published: dict[str, Any]) -> None:
    """Fleet pins have no provider kind, so a fleet-only resolve must stay silent.

    This is the shape `_SimRunnerProvider.default_policy` uses to build the baseline's mode table,
    and it runs on a machine with no producers installed — it must not trip the diagnostic.
    """
    fleet_only = ScenarioContent(
        fleet=(ContentPin(id="astro-mine.fleet.test-rover", reference=published["fleet_digest"]),)
    )
    resolved = ContentResolver(published["store"], provider_factories={}).resolve(fleet_only)
    assert resolved.unresolved == ()


def test_the_diagnostic_names_the_package_and_the_consequence(published: dict[str, Any]) -> None:
    """The operator-facing message: what is missing, what installs it, what it costs.

    Content and code ship separately — `astro-mine bench fetch` obtains the bundles, but rebuilding
    a world bundle into a `WorldProvider` is astro-mine-worlds' job — so a user who followed the
    documented quickstart can hold every digest and still have no physics.
    """
    resolved = ContentResolver(published["store"], provider_factories={}).resolve(
        _content(published)
    )

    message = describe_unresolved(resolved.unresolved)

    assert "astro-mine-worlds" in message
    assert "astro-mine-prospect" in message
    assert "astro-mine bench fetch" in message  # names the half the user already did
    assert "test-world" in message
