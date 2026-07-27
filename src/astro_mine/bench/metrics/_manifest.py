"""Core ``metric``-kind plugin manifests for Bench metrics (RM-P1-BENCH-12; bench.md §3, §6).

A Bench metric declares itself to the platform through a **Core plugin manifest**, never a
Bench-private schema — the same narrow-waist contract Studio negotiates its objective vocabulary
against (:meth:`astro_mine.studio.intent.validate.MetricVocabulary.from_manifest`, which reads a
``metric``-kind manifest's ``outputs`` as the declared metric keys) and the shape a community
metric plugin publishes to Hub so :mod:`astro_mine.bench.metrics._plugin` can discover it.

Two builders:

- :func:`reference_metric_manifest` — the manifest for the built-in reference set, its ``outputs``
  the seven reference metric keys. This is the *replaceable-example* vocabulary Studio consumes and
  the anchor scenario scores against (bench.md §2.4).
- :func:`metric_manifest` — the manifest for a single metric, its ``outputs`` that one key, its
  ``entrypoint`` attribute the ``module:attribute`` the reference loader materializes. This is how a
  community metric (or a test) authors a publishable ``metric`` artifact.

Core *describes* the manifest and never runs the metric (core.md §9); materialization is Bench's job
in :mod:`astro_mine.bench.metrics._plugin`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from astro_mine.bench.metrics._metric import Metric
from astro_mine.bench.metrics._reference import REFERENCE_METRICS
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.registry.model import ManifestDocument, PluginManifest

__all__ = [
    "REFERENCE_METRIC_MANIFEST_NAME",
    "manifest_document",
    "metric_manifest",
    "reference_metric_manifest",
]

#: The stable manifest name of the built-in reference metric set.
REFERENCE_METRIC_MANIFEST_NAME = "astro-mine.bench.reference-metrics"

#: The Core interfaces a Bench metric is built against: it consumes Observation traces
#: (``messages``) and reuses Core's objective→metric vocabulary (``objective``).
_METRIC_CORE_INTERFACES = {"objective": "0.1.0", "messages": "0.1.0"}


def _metric_attributes(metric: Metric) -> dict[str, str]:
    """The open-``attributes`` metadata Core does not schematize: a metric's unit + rules."""
    return {
        "unit": metric.unit,
        "direction": metric.direction.value,
        "aggregation": metric.aggregation.value,
    }


def metric_manifest(
    metric: Metric,
    *,
    name: str | None = None,
    entrypoint: str | None = None,
    core_interfaces: Mapping[str, str] | None = None,
) -> PluginManifest:
    """A ``metric``-kind :class:`PluginManifest` for a single metric plugin.

    ``outputs`` is the metric's one declared key (``metric.name``); ``version`` is the metric's own
    version, so the plugin loader can negotiate it. ``entrypoint`` (a ``module:attribute`` ref)
    rides in ``attributes`` for :func:`~astro_mine.bench.metrics.reference_metric_loader` to
    materialize; ``name`` defaults to the metric key but a publisher may namespace it (e.g.
    ``acme/comms-uptime``). This is the artifact a community metric publishes to Hub.
    """
    attributes = _metric_attributes(metric)
    if entrypoint is not None:
        attributes["entrypoint"] = entrypoint
    return PluginManifest(
        name=name or metric.name,
        version=metric.version,
        kind=PluginKind.METRIC,
        core_interfaces=dict(core_interfaces or _METRIC_CORE_INTERFACES),
        outputs=[metric.name],
        description=f"Bench metric {metric.name!r} ({metric.unit}, {metric.direction.value}).",
        license="Apache-2.0",
        attributes=attributes,
    )


def reference_metric_manifest(
    metrics: Sequence[Metric] = REFERENCE_METRICS,
) -> PluginManifest:
    """The ``metric``-kind manifest describing the built-in reference metric set (bench.md §2.4).

    Its ``outputs`` are the reference metric keys — the vocabulary Studio negotiates an objective's
    bindings against (:meth:`MetricVocabulary.from_manifest`) and the platform's shared definition
    of the reference measures (LUNAR-FR-009: the *same* metric definitions across Bench and Ops).
    The reference set ships as replaceable examples, so this manifest declares no single entrypoint.
    """
    return PluginManifest(
        name=REFERENCE_METRIC_MANIFEST_NAME,
        version="0.1.0",
        kind=PluginKind.METRIC,
        core_interfaces=dict(_METRIC_CORE_INTERFACES),
        outputs=[metric.name for metric in metrics],
        description="The Bench Phase-0 reference metric set (replaceable examples; bench.md §2.4).",
        license="Apache-2.0",
        attributes={
            "metric_versions": {metric.name: metric.version for metric in metrics},
            "units": {metric.name: metric.unit for metric in metrics},
        },
    )


def manifest_document(manifest: PluginManifest) -> ManifestDocument:
    """Wrap a :class:`PluginManifest` in a versioned :class:`ManifestDocument` for publishing."""
    return ManifestDocument(manifest_version="0.1", manifest=manifest)
