"""Hub-discovered community metric plugins (RM-P1-BENCH-12; bench.md §2.4, §3, §11).

A new measure of "good" for a swarm campaign is **a plugin, not a Bench code change**: the
community publishes a metric to Hub as a content-addressed OCI artifact carrying a Core
``metric``-kind plugin manifest, and Bench discovers it, verifies it fail-closed, and scores an
episode trace with it — the built-in reference set stays an untouched *replaceable example*
(bench.md §2.4, §3 extension points). This module is that discovery path, the metric-plugin twin
of the leaderboard's policy intake (:mod:`astro_mine.bench.leaderboard._hub`):

- :func:`resolve_metric_plugin` — resolve a Hub reference (a ``name:version`` tag or a ``sha256:``
  digest) to its verified :class:`~astro_mine.core.registry.PluginManifest`, **verifying twice**
  (the image manifest, its config, and every layer) before trusting a byte (hub.md verify-twice);
- :func:`validate_metric_manifest` — assert the manifest is a ``metric`` plugin that declares its
  metric keys as manifest ``outputs``, optionally satisfying a scenario's pinned Core interface;
- :func:`reference_metric_loader` / :func:`load_metric` — materialize the resolved artifact into a
  runnable :class:`~astro_mine.bench.metrics.Metric` behind an injected seam, then validate the
  loaded metric's metadata against what the manifest declared.

Bench imports **only Core + the Hub client**: the concrete :class:`astro_mine.hub.registry.Registry`
is opened lazily by :func:`open_registry` (behind the ``[leaderboard]`` extra), while this module
types the registry structurally (:class:`HubRegistry`) so it stays import-light and never reaches
into Sim or a private Hub schema (bench.md §2.2). Core *describes and resolves* a manifest but never
executes plugin code (core.md §9): materializing the metric object is Bench's job, here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from astro_mine.bench.metrics._metric import Metric, MetricError
from astro_mine.bench.metrics._registry import MetricRegistry
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.core.compat import check_compatible
from astro_mine.core.registry import PluginKind, load_plugin_manifest
from astro_mine.core.registry.model import PluginManifest

__all__ = [
    "HubRegistry",
    "MetricEntrypointError",
    "MetricManifestError",
    "MetricPluginError",
    "MetricPluginLoader",
    "ResolvedMetricPlugin",
    "discover_metric",
    "load_metric",
    "open_registry",
    "reference_metric_loader",
    "resolve_metric_plugin",
    "validate_metric_manifest",
]


class MetricPluginError(MetricError):
    """Raised when a Hub metric reference cannot be resolved, verified, or parsed to a manifest."""


class MetricManifestError(MetricError):
    """Raised when a manifest is not a metric plugin or fails its outputs/interface check."""


class MetricEntrypointError(MetricError):
    """Raised when a metric manifest's ``entrypoint`` does not load to a valid :class:`Metric`."""


class HubRegistry(Protocol):
    """The Hub-client surface Bench consumes — met by :class:`astro_mine.hub.registry.Registry`.

    Typed structurally so Bench depends on the Hub client only where a concrete registry is opened,
    keeping the discovery logic import-light and free of a private Hub schema (bench.md §2.2). It is
    the same surface the leaderboard's policy intake consumes.
    """

    def resolve(self, reference: str) -> Any:
        """Resolve a ``name:version`` tag or a digest to its manifest descriptor (``.digest``)."""
        ...

    def verify(self, digest: str) -> None:
        """Assert the stored blob at ``digest`` hashes to it — content-addressing on read."""
        ...

    def read_manifest(self, digest: str) -> dict[str, Any]:
        """The parsed OCI image manifest at ``digest`` (``config`` + ``layers`` descriptors)."""
        ...

    def read_config(self, manifest_digest: str) -> bytes:
        """The artifact's config blob (its Core plugin manifest) given the image-manifest digest."""
        ...


@dataclass(frozen=True)
class ResolvedMetricPlugin:
    """A Hub metric artifact resolved to its verified manifest + payload layer digests."""

    reference: str
    manifest_digest: str
    manifest: PluginManifest
    layer_digests: tuple[str, ...]


def resolve_metric_plugin(registry: HubRegistry, reference: str) -> ResolvedMetricPlugin:
    """Resolve ``reference`` from Hub to a verified :class:`ResolvedMetricPlugin` (bench.md §6, §9).

    Resolves the reference to its image-manifest digest, then **fail-closed verifies** the manifest,
    its config blob, and every payload layer against their content addresses before parsing the Core
    plugin manifest. Any resolution, integrity, or parse failure raises :class:`MetricPluginError` —
    a metric that cannot be authenticated by digest never scores anything.
    """
    try:
        descriptor = registry.resolve(reference)
    except Exception as exc:  # ArtifactNotFound and any client-specific resolution failure
        raise MetricPluginError(f"cannot resolve Hub reference {reference!r}: {exc}") from exc
    manifest_digest = str(descriptor.digest)
    try:
        # verify() re-checks the image manifest AND its config + every layer against their content
        # addresses in one call (hub.md §2.3 verify-twice) — fail-closed on any tampered blob.
        registry.verify(manifest_digest)
        image = registry.read_manifest(manifest_digest)
        layer_digests = tuple(str(layer["digest"]) for layer in image.get("layers", ()))
    except Exception as exc:
        raise MetricPluginError(f"integrity verification failed for {reference!r}: {exc}") from exc
    try:
        # The bare manifest, not a document — see the note on the same read in
        # `bench/leaderboard/_hub.py` (astro-mine-platform#14). A Hub-published metric plugin was
        # unreadable for exactly the reason a Hub-published policy was.
        manifest = load_plugin_manifest(registry.read_config(manifest_digest))
    except Exception as exc:
        raise MetricPluginError(f"invalid plugin manifest for {reference!r}: {exc}") from exc
    return ResolvedMetricPlugin(reference, manifest_digest, manifest, layer_digests)


def validate_metric_manifest(
    resolved: ResolvedMetricPlugin, spec: ScenarioSpec | None = None
) -> None:
    """Assert the artifact is a metric plugin that declares its keys (bench.md §3).

    The manifest must declare :attr:`PluginKind.METRIC` and name **at least one** metric key in its
    ``outputs`` (the keys Studio negotiates its objective vocabulary against —
    :meth:`MetricVocabulary.from_manifest`). When ``spec`` is given, the manifest's
    ``core_interfaces`` must additionally **satisfy** every Core interface the ScenarioSpec pins,
    under the same SemVer rule the registry applies at load. Raises :class:`MetricManifestError`
    otherwise — a mis-kinded or interface-incompatible metric is rejected before it runs.
    """
    manifest = resolved.manifest
    if manifest.kind is not PluginKind.METRIC:
        raise MetricManifestError(
            f"metric plugin {resolved.reference!r} must be kind=metric, got kind={manifest.kind}"
        )
    if not manifest.outputs:
        raise MetricManifestError(
            f"metric plugin {resolved.reference!r} declares no metric keys in manifest outputs"
        )
    if spec is not None:
        for name, required in spec.core_interface.items():
            provided = manifest.core_interfaces.get(name)
            if provided is None or not check_compatible(required, provided):
                raise MetricManifestError(
                    f"metric plugin {resolved.reference!r} interfaces {manifest.core_interfaces} "
                    f"do not satisfy scenario {spec.scenario_id!r} interface {name}>={required} "
                    f"(declared {provided!r})"
                )


class MetricPluginLoader(Protocol):
    """Materialize a resolved Hub metric plugin into a runnable :class:`Metric` (the seam).

    A loader whose metric lives in the artifact's *bytes* rather than in an importable module takes
    them from :func:`~astro_mine.bench.metrics.pull_verified_layer`, the one route Bench's registry
    seam offers: the layer is re-hashed against the digest the verified manifest commits to, and a
    digest that manifest does not commit to is refused (hub.md §2.3; conventions.md §9). A metric
    assembled from bytes no manifest vouched for scores nothing.
    """

    def __call__(self, resolved: ResolvedMetricPlugin, registry: HubRegistry) -> Metric:
        """Build the metric the scorer runs."""
        ...


def _resolve_entrypoint(reference: str, entrypoint: str) -> Metric:
    """Import a ``"module:attribute"`` metric reference and resolve it to a :class:`Metric`.

    The attribute may be a Metric instance, a Metric class, or a zero-arg factory (mirroring the
    policy loader). Raises :class:`MetricEntrypointError` on a malformed reference, an
    import/attribute failure, or an object that is not a Metric.
    """
    if ":" not in entrypoint:
        raise MetricEntrypointError(
            f"metric plugin {reference!r} entrypoint must be 'module:attribute', got {entrypoint!r}"
        )
    module_name, _, attribute = entrypoint.partition(":")
    try:
        obj = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise MetricEntrypointError(
            f"metric plugin {reference!r} entrypoint {entrypoint!r} did not import: {exc}"
        ) from exc
    candidate = obj() if isinstance(obj, type) or not hasattr(obj, "compute") else obj
    if not (hasattr(candidate, "compute") and callable(candidate.compute)):
        raise MetricEntrypointError(
            f"metric plugin {reference!r} entrypoint {entrypoint!r} is not a Metric (no compute)"
        )
    for attr in ("name", "version", "unit", "direction", "aggregation"):
        if not hasattr(candidate, attr):
            raise MetricEntrypointError(
                f"metric plugin {reference!r} entrypoint {entrypoint!r} is not a Metric "
                f"(missing {attr!r})"
            )
    return cast(Metric, candidate)


def reference_metric_loader(resolved: ResolvedMetricPlugin, registry: HubRegistry) -> Metric:
    """The dependency-clean default loader — resolve the manifest's ``entrypoint`` attribute.

    Reads a ``module:attribute`` :class:`Metric` reference from the manifest's ``entrypoint``
    attribute, imports it, and asserts the loaded metric's declared **name** is one it advertises in
    ``outputs`` and its **version** is compatible with the manifest version — so a metric cannot
    masquerade under a key it did not declare. A deployment can inject a different loader — e.g. one
    that materializes a metric from a payload layer taken from
    :func:`~astro_mine.bench.metrics.pull_verified_layer`, the verified route onto an artifact's
    bytes (hub.md §2.3). Raises :class:`MetricEntrypointError` when the manifest declares no
    entrypoint or the loaded object disagrees with the manifest.
    """
    entrypoint = resolved.manifest.attributes.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise MetricEntrypointError(
            f"metric plugin {resolved.reference!r} manifest declares no 'entrypoint' attribute; "
            "inject a MetricPluginLoader to materialize a payload-layer metric"
        )
    metric = _resolve_entrypoint(resolved.reference, entrypoint)
    if metric.name not in resolved.manifest.outputs:
        raise MetricEntrypointError(
            f"metric plugin {resolved.reference!r} loaded metric {metric.name!r} not in declared "
            f"outputs {resolved.manifest.outputs}"
        )
    if not check_compatible(metric.version, resolved.manifest.version):
        raise MetricEntrypointError(
            f"metric plugin {resolved.reference!r} metric version {metric.version} is incompatible "
            f"with manifest version {resolved.manifest.version}"
        )
    return metric


def load_metric(
    registry: HubRegistry,
    reference: str,
    *,
    spec: ScenarioSpec | None = None,
    loader: MetricPluginLoader = reference_metric_loader,
) -> Metric:
    """Resolve, verify, validate, and materialize a Hub metric plugin into a :class:`Metric`.

    The full discovery path in one call — the metric-plugin twin of the leaderboard's policy intake.
    ``spec`` (when given) additionally asserts the plugin's Core interfaces satisfy the scenario.
    Raises a :class:`MetricPluginError`, :class:`MetricManifestError`, or
    :class:`MetricEntrypointError` if any step fails.
    """
    resolved = resolve_metric_plugin(registry, reference)
    validate_metric_manifest(resolved, spec)
    return loader(resolved, registry)


def discover_metric(
    registry: HubRegistry,
    reference: str,
    into: MetricRegistry,
    *,
    spec: ScenarioSpec | None = None,
    loader: MetricPluginLoader = reference_metric_loader,
    replace: bool = False,
) -> Metric:
    """Load a Hub metric plugin and overlay it on a :class:`MetricRegistry` — no Bench code change.

    The end-to-end acceptance path (bench.md §3): a community metric published to Hub is discovered,
    verified, and registered onto ``into`` so a scenario resolves and scores against it **without
    editing Bench's built-in catalog**. Returns the registered metric.
    """
    metric = load_metric(registry, reference, spec=spec, loader=loader)
    return into.register(metric, replace=replace)


def open_registry(path: str | Path) -> HubRegistry:
    """Open the content-addressed Hub registry at ``path`` (requires the ``[leaderboard]`` extra).

    Thin lazy wrapper over :class:`astro_mine.hub.registry.Registry` so the base package imports
    without the Hub client; ``path`` is the workspace tier-1 registry (the ``files/hub-registry``
    convention).
    """
    from astro_mine.hub.registry import Registry

    return Registry(path)
