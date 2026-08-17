"""One config-blob convention, asserted against the platform's own publisher (platform#14).

**The bug this exists to make impossible was invisible to every component suite, and that is the
point of putting it here.** Bench's Hub-digest intake read the artifact config as a
:class:`ManifestDocument` envelope; every publisher writes a bare
:class:`~astro_mine.core.registry.PluginManifest`. Both sides were internally consistent and
thoroughly tested — Bench's fixtures *published the envelope themselves* — so the whole community
submission path (RM-P1-BENCH-10) could not accept a single real artifact while the suite stayed
green. It surfaced only when a deployment was seeded for real (astro-mine-api#14).

A cross-component disagreement needs a cross-component test. Two rules, and neither can live in one
component's directory:

1. **What the platform publishes, every reader in the platform can read.** The subject is produced
   by :meth:`HubClient.publish` — the real publisher, through the real admission gate — not by a
   fixture's own call to ``registry.publish``. A fixture that constructs the bytes it is about to
   parse asserts that a function is its own inverse and nothing else, which is exactly how this
   defect survived.
2. **There is one convention, not two.** The envelope must be *rejected*, so a future reader cannot
   quietly reintroduce the second shape and take the registry with it — one envelope-shaped
   artifact makes ``catalog_from_registry`` raise, and Studio's asset and world menus go with it.

The reader shapes below are enumerated from the source rather than described. There are exactly two
in the platform, and both are checked against the same published blob:

- ``PluginManifest.model_validate_json(...)`` — Hub's client and catalog, Fleet, Prospect,
  Surrogate, Studio (eight call sites);
- ``load_plugin_manifest(...)`` — Bench's two intakes, which additionally need the validation the
  bare Pydantic call does not perform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.bench.leaderboard import resolve_submission
from astro_mine.bench.metrics._plugin import resolve_metric_plugin
from astro_mine.core.registry import ManifestValidationError, load_plugin_manifest
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.model import ManifestDocument, PluginManifest
from astro_mine.hub.client import HubClient, catalog_from_registry
from astro_mine.hub.registry import Blob, Registry
from astro_mine.hub.supply_chain import generate_keypair

ONNX_MEDIA_TYPE = "application/vnd.astro-mine.policy.onnx.v1"


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "hub-registry")


def _manifest(
    name: str = "lawnmower-survey",
    version: str = "0.2.0",
    kind: PluginKind = PluginKind.POLICY,
) -> PluginManifest:
    return PluginManifest(
        name=name,
        version=version,
        kind=kind,
        core_interfaces={"observation": "0.1.0"},
        inputs=["Observation"],
        outputs=["ActionBatch"],
        attributes={"entrypoint": "tests.bench._factories:BASELINE_INSTANCE"},
    )


def _publish(registry: Registry, manifest: PluginManifest, *, kind: str = "policy") -> str:
    """Publish through :class:`HubClient` — the publisher every producer in the platform uses.

    Deliberately not ``registry.publish``: the defect this file guards lived in the gap between what
    ``HubClient.publish`` writes and what a fixture wrote for itself, so a fixture that reached past
    the client would reproduce the gap rather than close it.
    """
    private_pem, _ = generate_keypair()
    published = HubClient(registry).publish(
        name=manifest.name,
        version=manifest.version,
        kind=kind,
        manifest=manifest,
        layers=[Blob(ONNX_MEDIA_TYPE, b"onnx-model-bytes")],
        private_key_pem=private_pem,
    )
    return str(published.digest)


# --- rule 1: what the platform publishes, the platform can read -------------------------------


def test_the_config_blob_is_the_bare_manifest(registry: Registry) -> None:
    """The shape itself, named — every assertion below depends on this one being true."""
    digest = _publish(registry, _manifest())
    config = json.loads(registry.read_config(digest))

    # hub.md §2 principle 2: "Hub indexes artifacts by the Core plugin manifest." The manifest's
    # own fields are at the top level; there is no envelope to reach through.
    assert config["name"] == "lawnmower-survey"
    assert config["kind"] == "policy"
    assert "manifest_version" not in config
    assert "manifest" not in config


def test_every_reader_shape_in_the_platform_reads_it(registry: Registry) -> None:
    """Both read shapes, against one published blob.

    Enumerated rather than described: if a third convention is ever introduced, it belongs in this
    test, and adding it should feel like the decision it is.
    """
    digest = _publish(registry, _manifest())
    config = registry.read_config(digest)

    # The eight-call-site shape — Hub's client and catalog, Fleet, Prospect, Surrogate, Studio.
    assert PluginManifest.model_validate_json(config).name == "lawnmower-survey"

    # Bench's shape, which parses the same bytes *and* validates them.
    assert load_plugin_manifest(config).name == "lawnmower-survey"


def test_bench_hub_intake_accepts_a_published_policy(registry: Registry) -> None:
    """The reported reproducer, at the platform level.

    ``POST /bench/submissions/hub`` answered 404 ``content_not_found`` for every artifact the
    platform publishes, with a schema error naming every real manifest field as unexpected. The
    route is astro-mine-api's; the resolution underneath it is this call.
    """
    manifest = _manifest()
    digest = _publish(registry, manifest)

    resolved = resolve_submission(registry, digest)

    assert resolved.manifest.name == manifest.name
    assert resolved.manifest.kind is PluginKind.POLICY
    assert resolved.layer_digests  # the ONNX payload travelled with it


def test_bench_metric_plugin_discovery_accepts_a_published_metric(registry: Registry) -> None:
    """The same defect through RM-P1-BENCH-12 — a community metric was equally unreadable."""
    manifest = _manifest(name="coverage", kind=PluginKind.METRIC)
    digest = _publish(registry, manifest, kind="plugin")

    resolved = resolve_metric_plugin(registry, digest)

    assert resolved.manifest.name == "coverage"
    assert resolved.manifest.kind is PluginKind.METRIC


def test_a_published_artifact_leaves_the_registry_catalogable(registry: Registry) -> None:
    """``catalog_from_registry`` iterates *every* reference, so one bad artifact breaks all of them.

    This is why the envelope was never a viable workaround: publishing it would have satisfied Bench
    and made the whole registry unreadable to Studio's asset and world menus.
    """
    _publish(registry, _manifest())
    _publish(
        registry, _manifest(name="coverage", kind=PluginKind.METRIC), kind="plugin"
    )

    catalog = catalog_from_registry(registry)

    assert {record.manifest.name for record in catalog.all()} == {
        "lawnmower-survey",
        "coverage",
    }


# --- rule 2: one convention, not two ----------------------------------------------------------


def test_the_envelope_is_rejected_and_says_which_shape_it_got() -> None:
    """A document handed to the config-blob reader fails, and fails *legibly*.

    Pydantic's own report for this is five missing required fields, which reads like a corrupt
    manifest rather than a well-formed one at the wrong level — and misreading it that way is how a
    reader concludes the *artifact* is broken. The whole issue is that these two shapes are easy to
    confuse, so the error names them.
    """
    envelope = ManifestDocument(manifest_version="0.1", manifest=_manifest())
    source = json.dumps(envelope.model_dump(mode="json"))

    with pytest.raises(ManifestValidationError, match="manifest \\*document\\*"):
        load_plugin_manifest(source)


def test_validation_is_not_lost_on_the_community_intake_path() -> None:
    """The gated-capability-tag gate still fires — the check the naive fix would have deleted.

    Reading the config with a bare ``PluginManifest.model_validate_json`` would have fixed the
    parse and silently dropped this, on the one path where a *third party's* manifest arrives
    (conventions.md §12). That trade is the reason ``load_plugin_manifest`` exists rather than a
    one-line swap in two files.
    """
    gated = json.loads(_manifest().model_dump_json())
    gated["capability_tags"] = ["operational_targeting"]

    with pytest.raises(ManifestValidationError, match="reserved/gated capability tag"):
        load_plugin_manifest(json.dumps(gated))
