"""Hub-discovered community metric plugins (RM-P1-BENCH-12; bench.md §2.4, §3, §11).

The acceptance criterion: **a community metric plugin published to Hub is discovered and scores an
episode trace with no Bench code change.** These tests publish an ``earth_contact_uptime`` metric —
deliberately *not* one of the seven reference metrics — to a tier-1 Hub registry, then discover it,
score a trace with it, and assert the built-in registry was never touched. The verify-twice
fail-closed and manifest-validation negatives mirror the leaderboard's policy intake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.bench.metrics import (
    REFERENCE_METRICS,
    REGISTRY,
    MetricEntrypointError,
    MetricManifestError,
    MetricPluginError,
    discover_metric,
    load_metric,
    manifest_document,
    metric_manifest,
    reference_metric_manifest,
    reference_registry,
    resolve_metric_plugin,
)
from astro_mine.bench.scenario import MetricRef
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.model import PluginManifest
from astro_mine.hub.registry import Blob, Registry
from astro_mine.hub.registry._oci import blob_path
from tests.bench._factories import COMMUNITY_METRIC, make_observation, make_trace

COMMUNITY_ENTRYPOINT = "tests.bench._factories:COMMUNITY_METRIC"
COMMUNITY_KEY = "earth_contact_uptime"
METRIC_MEDIA_TYPE = "application/vnd.astro-mine.metric.payload.v1"
METRIC_PAYLOAD = b"metric-payload-bytes"


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "hub-registry")


def _publish(
    registry: Registry,
    *,
    name: str = "acme/earth-contact-uptime",
    version: str = "0.1.0",
    manifest: PluginManifest | None = None,
) -> str:
    """Publish a metric plugin (config = a Core ``metric`` manifest) and return its digest.

    A metric artifact rides the generic ``plugin`` OCI artifact kind; the *fine-grained* kind lives
    in the Core manifest (``PluginKind.METRIC``), the layering Hub indexes by. The payload layer is
    what a metric that does *not* ship as an importable module lives in — the reference loader never
    reads it, but intake verifies it fail-closed like any other blob (hub.md §2.3).
    """
    manifest = manifest or metric_manifest(
        COMMUNITY_METRIC, name=name, entrypoint=COMMUNITY_ENTRYPOINT
    )
    published = registry.publish(
        name=name,
        version=version,
        kind="plugin",
        config=manifest_document(manifest).model_dump(mode="json"),
        layers=[Blob(METRIC_MEDIA_TYPE, METRIC_PAYLOAD)],
    )
    return published.digest


def _uptime_trace() -> object:
    """A trace with 3 of 4 ticks in Earth contact → an uptime of 0.75."""
    return make_trace(
        [
            make_observation(0, 0.0, earth_contact=True),
            make_observation(1, 60.0, earth_contact=True),
            make_observation(2, 120.0, earth_contact=False),
            make_observation(3, 180.0, earth_contact=True),
        ]
    )


def test_community_metric_is_not_a_reference_metric() -> None:
    # The premise: earth_contact_uptime is genuinely new — not in the built-in catalog.
    assert COMMUNITY_KEY not in REGISTRY
    assert COMMUNITY_KEY not in {m.name for m in REFERENCE_METRICS}
    assert COMMUNITY_KEY not in reference_registry()


def test_discovered_metric_scores_a_trace_with_no_bench_code_change(registry: Registry) -> None:
    digest = _publish(registry)
    # Discover the Hub metric onto a fresh registry — the built-in catalog is never edited.
    catalog = reference_registry()
    discover_metric(registry, digest, into=catalog)

    assert COMMUNITY_KEY in catalog.names
    metric = catalog.resolve(MetricRef(name=COMMUNITY_KEY, version="0.1.0"))
    assert metric.compute(_uptime_trace()).value == pytest.approx(0.75)

    # The acceptance guarantee: no Bench code change — the module-level registry is untouched.
    assert COMMUNITY_KEY not in REGISTRY


def test_load_metric_materializes_the_metric(registry: Registry) -> None:
    metric = load_metric(registry, _publish(registry))
    assert metric.name == COMMUNITY_KEY
    assert metric.compute(_uptime_trace()).value == pytest.approx(0.75)


def test_resolve_by_name_version_tag(registry: Registry) -> None:
    _publish(registry, name="acme/earth-contact-uptime")
    resolved = resolve_metric_plugin(registry, "acme/earth-contact-uptime:0.1.0")
    assert resolved.manifest.kind is PluginKind.METRIC
    assert resolved.manifest.outputs == [COMMUNITY_KEY]


def test_scenario_interface_check(registry: Registry) -> None:
    # With a spec, the plugin's declared interfaces must satisfy the scenario's pinned ones.
    anchor = load_scenario(ANCHOR_SCENARIO_ID)
    manifest = metric_manifest(
        COMMUNITY_METRIC,
        name="acme/uptime",
        entrypoint=COMMUNITY_ENTRYPOINT,
        core_interfaces={"objective": "9.9.9"},  # does not satisfy the anchor
    )
    digest = _publish(registry, name="acme/uptime", manifest=manifest)
    with pytest.raises(MetricManifestError, match="do not satisfy"):
        load_metric(registry, digest, spec=anchor)


def test_wrong_kind_is_rejected(registry: Registry) -> None:
    manifest = PluginManifest(
        name="acme/not-a-metric",
        version="0.1.0",
        kind=PluginKind.POLICY,
        outputs=[COMMUNITY_KEY],
        attributes={"entrypoint": COMMUNITY_ENTRYPOINT},
    )
    digest = _publish(registry, name="acme/not-a-metric", manifest=manifest)
    with pytest.raises(MetricManifestError, match="must be kind=metric"):
        load_metric(registry, digest)


def test_manifest_without_declared_outputs_is_rejected(registry: Registry) -> None:
    manifest = PluginManifest(
        name="acme/empty",
        version="0.1.0",
        kind=PluginKind.METRIC,
        outputs=[],
        attributes={"entrypoint": COMMUNITY_ENTRYPOINT},
    )
    digest = _publish(registry, name="acme/empty", manifest=manifest)
    with pytest.raises(MetricManifestError, match="declares no metric keys"):
        load_metric(registry, digest)


def test_missing_entrypoint_is_rejected(registry: Registry) -> None:
    manifest = PluginManifest(
        name="acme/no-entry", version="0.1.0", kind=PluginKind.METRIC, outputs=[COMMUNITY_KEY]
    )
    digest = _publish(registry, name="acme/no-entry", manifest=manifest)
    with pytest.raises(MetricEntrypointError, match="no 'entrypoint'"):
        load_metric(registry, digest)


def test_entrypoint_key_must_match_declared_output(registry: Registry) -> None:
    # The loaded metric declares 'earth_contact_uptime' but the manifest advertises another key.
    manifest = PluginManifest(
        name="acme/mismatch",
        version="0.1.0",
        kind=PluginKind.METRIC,
        outputs=["some_other_key"],
        attributes={"entrypoint": COMMUNITY_ENTRYPOINT},
    )
    digest = _publish(registry, name="acme/mismatch", manifest=manifest)
    with pytest.raises(MetricEntrypointError, match="not in declared outputs"):
        load_metric(registry, digest)


def test_unknown_reference_fails_closed(registry: Registry) -> None:
    with pytest.raises(MetricPluginError, match="cannot resolve"):
        resolve_metric_plugin(registry, "sha256:" + "0" * 64)


def test_tampered_config_fails_closed(registry: Registry, tmp_path: Path) -> None:
    digest = _publish(registry)
    config_hex = registry.read_manifest(digest)["config"]["digest"].split(":", 1)[1]
    blob = tmp_path / "hub-registry" / "blobs" / "sha256" / config_hex
    blob.write_bytes(b"tampered-manifest-bytes")
    with pytest.raises(MetricPluginError, match="integrity verification failed"):
        resolve_metric_plugin(registry, digest)


def test_tampered_payload_layer_fails_closed(registry: Registry) -> None:
    # The twin of the leaderboard's tampered-layer regression: intake verifies *every* blob the
    # manifest commits to, not just the config it parses. A metric whose payload no longer hashes to
    # its content address is refused before it can score anything (bench.md §9; hub.md §2.3).
    digest = _publish(registry)
    layer_digest = registry.read_manifest(digest)["layers"][0]["digest"]
    blob_path(registry.path, layer_digest).write_bytes(b"tampered-payload-bytes")
    with pytest.raises(MetricPluginError, match="integrity verification failed"):
        resolve_metric_plugin(registry, digest)


def test_reference_manifest_declares_every_reference_key() -> None:
    # The vocabulary Studio negotiates against (MetricVocabulary.from_manifest reads outputs).
    manifest = reference_metric_manifest()
    assert manifest.kind is PluginKind.METRIC
    assert set(manifest.outputs) == {m.name for m in REFERENCE_METRICS}
