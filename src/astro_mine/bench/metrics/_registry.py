"""Metric registry + version negotiation.

Resolves the :class:`~astro_mine.bench.scenario.MetricRef` names a ScenarioSpec pins to the
concrete metric plugins that score it, negotiating the requested interface version against the
registered one with Core's exact rule (:func:`astro_mine.core.compat.check_compatible`): a
pre-1.0 ``0.y`` metric requires an *exact* minor.

The built-in reference set (:data:`REGISTRY`) ships as *replaceable examples* (bench.md §2.4).
A :class:`MetricRegistry` overlays **community metric plugins discovered via Hub** on top of that
base set, so a new measure of "good" scores a scenario with **no Bench code change** (bench.md §3,
§11) — the discovery + verification path lives in :mod:`astro_mine.bench.metrics._plugin`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from astro_mine.bench.metrics._metric import Metric, MetricError
from astro_mine.bench.metrics._reference import REFERENCE_METRICS
from astro_mine.bench.scenario import MetricRef
from astro_mine.core.compat import check_compatible

__all__ = [
    "REGISTRY",
    "DuplicateMetric",
    "IncompatibleMetricVersion",
    "MetricRegistry",
    "UnknownMetric",
    "reference_registry",
    "resolve_metric",
    "resolve_metrics",
]

#: The built-in reference metrics, keyed by name.
REGISTRY: dict[str, Metric] = {metric.name: metric for metric in REFERENCE_METRICS}


class UnknownMetric(MetricError):
    """Raised when a scenario references a metric name that is not registered."""


class IncompatibleMetricVersion(MetricError):
    """Raised when a scenario's requested metric version is not satisfied by the registry."""


class DuplicateMetric(MetricError):
    """Raised when a metric name is registered twice without ``replace=True``."""


def _negotiate(name: str, requested: str, provided: Metric) -> Metric:
    """Apply Core's version rule; raise :class:`IncompatibleMetricVersion` on a mismatch."""
    if not check_compatible(requested, provided.version):
        raise IncompatibleMetricVersion(
            f"metric {name!r}: scenario requires {requested}, registry provides {provided.version}"
        )
    return provided


class MetricRegistry:
    """A resolvable metric set — the built-in reference metrics overlaid with discovered plugins.

    Seeded from the reference set by default, so a fresh registry resolves exactly what the
    hardcoded :data:`REGISTRY` does. :meth:`register` overlays a **community metric plugin** (loaded
    from Hub by :mod:`astro_mine.bench.metrics._plugin`) without mutating the module-level builtins,
    so discovering a new metric is never a code change to the reference catalog (bench.md §3).
    """

    def __init__(self, metrics: Iterable[Metric] | None = None) -> None:
        base = REFERENCE_METRICS if metrics is None else metrics
        self._by_name: dict[str, Metric] = {metric.name: metric for metric in base}

    def register(self, metric: Metric, *, replace: bool = False) -> Metric:
        """Overlay ``metric`` on the registry; raise :class:`DuplicateMetric` on a name clash.

        A discovered community plugin (or a locally authored metric) is added here. ``replace=True``
        shadows a built-in of the same name — the seam a scenario uses to pin a community reworking
        of a reference metric. Returns the registered metric for call chaining.
        """
        if not replace and metric.name in self._by_name:
            raise DuplicateMetric(
                f"metric {metric.name!r} already registered; pass replace=True to override"
            )
        self._by_name[metric.name] = metric
        return metric

    def resolve(self, ref: MetricRef) -> Metric:
        """Resolve one :class:`MetricRef`, negotiating its version against the registered metric."""
        metric = self._by_name.get(ref.name)
        if metric is None:
            raise UnknownMetric(
                f"no metric registered under {ref.name!r} (known: {sorted(self._by_name)})"
            )
        return _negotiate(ref.name, ref.version, metric)

    def resolve_all(self, refs: Sequence[MetricRef]) -> tuple[Metric, ...]:
        """Resolve a scenario's metric references, in order (e.g. ``spec.metrics``)."""
        return tuple(self.resolve(ref) for ref in refs)

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered metric name, in stable sorted order."""
        return tuple(sorted(self._by_name))

    def __contains__(self, name: object) -> bool:
        return name in self._by_name


def reference_registry() -> MetricRegistry:
    """A fresh :class:`MetricRegistry` seeded with only the built-in reference metrics."""
    return MetricRegistry()


def resolve_metric(ref: MetricRef) -> Metric:
    """Resolve one :class:`MetricRef` against the built-in reference set, negotiating the version.

    The built-in path (community plugins are resolved through a :class:`MetricRegistry`). Raises
    :class:`UnknownMetric` if the name is not registered, or :class:`IncompatibleMetricVersion` if
    the registered version cannot satisfy the request.
    """
    metric = REGISTRY.get(ref.name)
    if metric is None:
        raise UnknownMetric(f"no metric registered under {ref.name!r} (known: {sorted(REGISTRY)})")
    return _negotiate(ref.name, ref.version, metric)


def resolve_metrics(refs: Sequence[MetricRef]) -> tuple[Metric, ...]:
    """Resolve a scenario's metric references, in order (e.g. ``spec.metrics``)."""
    return tuple(resolve_metric(ref) for ref in refs)
