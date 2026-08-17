# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry + Prometheus for the submission pipeline (bench#32; bench.md §10; CX-OBS).

bench.md §10 asks for two things and this module is both:

- **Telemetry.** *"OpenTelemetry traces/metrics/logs across submit → evaluate → score → rank, so a
  leaderboard entry is traceable end-to-end."* :func:`span` instruments each stage, and
  :func:`inject_trace_context` / :func:`extract_trace_context` carry the **W3C traceparent** across
  the pipeline's asynchronous hop (the submission is accepted, queued, and picked up by a worker —
  NATS/JetStream in the deployment), so the trace does not break at the queue.
- **Metrics & dashboards.** *"Prometheus + Grafana for queue depth, evaluation throughput,
  **re-execution mismatch rate (a key integrity signal)**, and per-scenario cost."*
  :class:`PipelineMetrics` exposes exactly those as Prometheus series, and
  :func:`metrics_exposition` renders them for the app's ``/metrics`` endpoint. The Grafana starter
  that reads them ships with the REST tier that serves that endpoint — ``deploy/grafana/`` in
  ``astro-mine-api``, where a test asserts its PromQL names the series below.

**The base package stays ``core + pydantic``.** Neither ``opentelemetry`` nor ``prometheus_client``
is imported at module scope: both are probed once and fall back to no-op shims, so
``import astro_mine.bench`` — and the whole offline local tier — works with neither installed, and
instrumented code needs no ``if telemetry_enabled:`` guards. The real SDKs arrive with the
``[observability]`` extra, which a hosted deployment installs.

Backlog: bench#32 — astro-mine-bench#32
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

__all__ = [
    "PROMETHEUS_CONTENT_TYPE",
    "PipelineMetrics",
    "current_trace_id",
    "extract_trace_context",
    "inject_trace_context",
    "metrics",
    "metrics_exposition",
    "otel_available",
    "prometheus_available",
    "span",
]

#: The exposition format ``GET /metrics`` serves (Prometheus text format v0.0.4).
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: The OTel instrumentation scope for every span Bench emits.
_SCOPE = "astro_mine.bench"

#: The Prometheus metric-name prefix (conventions.md §10; bench.md §10).
_PREFIX = "astro_mine_bench"


def otel_available() -> bool:
    """Whether the OpenTelemetry API is importable (the ``[observability]`` extra)."""
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError:
        return False
    return True


def prometheus_available() -> bool:
    """Whether ``prometheus_client`` is importable (the ``[observability]`` extra)."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True


# --- tracing: submit -> evaluate -> score -> rank -------------------------------------------------


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open an OTel span named ``name`` for one pipeline stage; a no-op without the SDK.

    Stage names are the pipeline's own vocabulary — ``bench.submit``, ``bench.verify``,
    ``bench.evaluate``, ``bench.score``, ``bench.rank`` — so a leaderboard entry reads end-to-end as
    one trace (bench.md §10). Exceptions are recorded on the span and re-raised: a rejected
    submission is *visible* in the trace, not swallowed.
    """
    if not otel_available():
        yield None
        return
    import opentelemetry.trace as trace_api

    tracer = trace_api.get_tracer(_SCOPE)
    with tracer.start_as_current_span(name) as active:
        for key, value in attributes.items():
            if value is not None:
                active.set_attribute(key, value)
        try:
            yield active
        except Exception as exc:
            active.record_exception(exc)
            active.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exc)))
            raise


def current_trace_id() -> str | None:
    """The active span's 32-hex trace id, for stamping onto audit records; ``None`` if untraced."""
    if not otel_available():
        return None
    import opentelemetry.trace as trace_api

    context = trace_api.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


def inject_trace_context(carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Write the active trace context into ``carrier`` as W3C ``traceparent``/``tracestate``.

    The carrier is the message-header map on the submission's async hop — the queue envelope here,
    NATS/JetStream headers in the deployment (bench.md §10: *"trace IDs propagated through async
    hops"*). A no-op without the SDK, so the pipeline is identical either way.
    """
    if otel_available():
        from opentelemetry.propagate import inject

        inject(carrier)
    return carrier


@contextmanager
def extract_trace_context(carrier: Mapping[str, str]) -> Iterator[None]:
    """Re-attach the trace context from ``carrier`` for the consumer side of the async hop.

    Spans opened inside this block are children of the *producer's* span, across the queue — which
    is what makes the whole submit → evaluate → score → rank pipeline a single trace even though a
    worker, not the request handler, does most of it.
    """
    if not otel_available():
        yield
        return
    from opentelemetry import context as context_api
    from opentelemetry.propagate import extract

    token = context_api.attach(extract(dict(carrier)))
    try:
        yield
    finally:
        context_api.detach(token)


# --- metrics: queue depth, mismatch rate, evaluation latency --------------------------------------


class _NoopMetric:
    """The shim every Prometheus instrument degrades to when the client is not installed."""

    def labels(self, **_: str) -> _NoopMetric:
        return self

    def inc(self, amount: float = 1.0) -> None: ...

    def dec(self, amount: float = 1.0) -> None: ...

    def set(self, value: float) -> None: ...

    def observe(self, value: float) -> None: ...


class PipelineMetrics:
    """The Prometheus series bench.md §10 names, plus the ones bench#29/#30 made measurable.

    - :attr:`queue_depth` — submissions accepted but not yet evaluated (**queue depth**);
    - :attr:`reexecutions` — labelled ``verdict=verified|mismatch``: the **re-execution mismatch
      rate**, bench.md §10's "key integrity signal", is the ratio of the two;
    - :attr:`evaluation_seconds` — a histogram per pipeline stage: **evaluation latency** and, by
      its ``_count`` rate, evaluation throughput;
    - :attr:`submissions` — labelled by scenario and terminal outcome (ranked/flagged/rejected);
    - :attr:`authz_decisions` and :attr:`verifications` — the authorization and supply-chain signals
      from bench#29; a spike in denials or verification failures is an attack indicator;
    - :attr:`sandbox_terminations` — labelled by :class:`~astro_mine.bench.sandbox.SandboxStatus`:
      how often a submission is killed by its envelope (bench#30).

    Instruments are created against a caller-supplied registry so a test can assert on an isolated
    one; the module-level :func:`metrics` singleton is what the app uses.
    """

    def __init__(self, registry: Any = None) -> None:
        if not prometheus_available():
            self.registry: Any = None
            noop = _NoopMetric()
            self.queue_depth: Any = noop
            self.submissions: Any = noop
            self.reexecutions: Any = noop
            self.evaluation_seconds: Any = noop
            self.authz_decisions: Any = noop
            self.verifications: Any = noop
            self.sandbox_terminations: Any = noop
            return

        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = registry if registry is not None else CollectorRegistry()
        self.queue_depth = Gauge(
            f"{_PREFIX}_queue_depth",
            "Submissions accepted and awaiting evaluation.",
            registry=self.registry,
        )
        self.submissions = Counter(
            f"{_PREFIX}_submissions_total",
            "Submissions by scenario and terminal outcome.",
            ("scenario", "outcome"),
            registry=self.registry,
        )
        self.reexecutions = Counter(
            f"{_PREFIX}_reexecutions_total",
            "Sampled provenance re-executions by verdict; the mismatch rate is the key integrity "
            "signal (bench.md §10).",
            ("scenario", "verdict"),
            registry=self.registry,
        )
        self.evaluation_seconds = Histogram(
            f"{_PREFIX}_evaluation_duration_seconds",
            "Wall-clock duration of each submission-pipeline stage.",
            ("scenario", "stage"),
            registry=self.registry,
        )
        self.authz_decisions = Counter(
            f"{_PREFIX}_authz_decisions_total",
            "Authorization decisions by action and outcome (bench#29).",
            ("action", "decision"),
            registry=self.registry,
        )
        self.verifications = Counter(
            f"{_PREFIX}_supply_chain_verifications_total",
            "Submission supply-chain verifications by outcome (bench#29).",
            ("outcome",),
            registry=self.registry,
        )
        self.sandbox_terminations = Counter(
            f"{_PREFIX}_sandbox_terminations_total",
            "Sandboxed submission runs by terminal status (bench#30).",
            ("status",),
            registry=self.registry,
        )

    @contextmanager
    def time_stage(self, *, scenario: str, stage: str) -> Iterator[None]:
        """Time one pipeline stage into :attr:`evaluation_seconds` — success or failure alike."""
        import time

        started = time.monotonic()
        try:
            yield
        finally:
            self.evaluation_seconds.labels(scenario=scenario, stage=stage).observe(
                time.monotonic() - started
            )

    def exposition(self) -> bytes:
        """Render the registry in Prometheus text format (what ``GET /metrics`` returns)."""
        if self.registry is None:
            return b"# prometheus_client is not installed; install the [observability] extra\n"
        from prometheus_client import generate_latest

        rendered: bytes = generate_latest(self.registry)
        return rendered


_METRICS: PipelineMetrics | None = None


def metrics() -> PipelineMetrics:
    """The process-wide :class:`PipelineMetrics` the app and service instrument against."""
    global _METRICS
    if _METRICS is None:
        _METRICS = PipelineMetrics()
    return _METRICS


def metrics_exposition() -> tuple[bytes, str]:
    """The body + content type for the leaderboard's Prometheus ``/metrics`` endpoint."""
    return metrics().exposition(), PROMETHEUS_CONTENT_TYPE
