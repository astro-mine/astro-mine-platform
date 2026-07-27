"""SafetyVerdict — the auditable output surface of the Guard shield (RM-P1-GUARD-06).

Every tick's :class:`SafetyVerdict` — the certified action, whether/why an intervention occurred,
the invoked spec clause(s), the active layer, and the barrier-margin certificate — carrying the
spec/model content-hash + Guard-code-version provenance a safety claim needs (guard.md §5, §6). The
verdict stream is written to **MCAP** for channel-by-channel replay and aggregated to **Parquet**
so [Bench](https://github.com/astro-mine/astro-mine-bench) scores "violations per scenario" and the
"performance cost of shielding," and [View] can overlay interventions.

Two hard rules (guard.md §6, §8, §9.1):

- **The verdict only records; it never decides.** It mirrors the fail-safe decision the trusted
  Rust safety core already made — a logging fault never changes the certified action.
- **Telemetry is best-effort and off the safety path.** The MCAP/Parquet writers (behind the
  ``[recording]`` / ``[metrics]`` optional extras) are imported lazily and swallow their faults, so
  the base package stays core + pydantic and back-pressure never stalls the shield.

Public API:

- the message — :class:`SafetyVerdict` (+ :data:`VERDICT_VERSION`, :func:`load_schema`) and its
  Protobuf wire form (:func:`verdict_to_wire` / :func:`verdict_from_wire` / ``*_proto``);
- Core-catalog registration — :func:`build_verdict_manifest` / :func:`register_verdict_schema`;
- the sinks — :class:`VerdictSink`, :class:`CollectingSink`, :class:`NullSink`, and the MCAP
  :class:`VerdictStream` (:func:`open_verdict_stream` / :func:`read_verdicts`);
- Bench metrics — :func:`violations_per_scenario`, :func:`shielding_cost`,
  :func:`write_metrics_parquet` (+ :func:`verdicts_to_table`).
"""

from __future__ import annotations

from astro_mine.guard.audit.catalog import (
    SAFETY_VERDICT_INTERFACE_VERSIONS,
    SAFETY_VERDICT_OUTPUT,
    build_verdict_manifest,
    register_verdict_schema,
    verdict_schema_content_hash,
)
from astro_mine.guard.audit.metrics import (
    FALLBACK_INTERVENTION,
    NO_INTERVENTION,
    VIOLATION_REASONS,
    ShieldingCost,
    ViolationCounts,
    shielding_cost,
    verdicts_to_table,
    violations_per_scenario,
    write_metrics_parquet,
)
from astro_mine.guard.audit.model import VERDICT_VERSION, SafetyVerdict, load_schema
from astro_mine.guard.audit.sink import CollectingSink, NullSink, VerdictSink
from astro_mine.guard.audit.stream import (
    VERDICT_SCHEMA_NAME,
    VERDICTS_TOPIC,
    VerdictStream,
    open_verdict_stream,
    read_verdicts,
)
from astro_mine.guard.audit.wire import (
    verdict_from_proto,
    verdict_from_wire,
    verdict_to_proto,
    verdict_to_wire,
)

__all__ = [
    "FALLBACK_INTERVENTION",
    "NO_INTERVENTION",
    "SAFETY_VERDICT_INTERFACE_VERSIONS",
    "SAFETY_VERDICT_OUTPUT",
    "VERDICTS_TOPIC",
    "VERDICT_SCHEMA_NAME",
    "VERDICT_VERSION",
    "VIOLATION_REASONS",
    "CollectingSink",
    "NullSink",
    "SafetyVerdict",
    "ShieldingCost",
    "VerdictSink",
    "VerdictStream",
    "ViolationCounts",
    "build_verdict_manifest",
    "load_schema",
    "open_verdict_stream",
    "read_verdicts",
    "register_verdict_schema",
    "shielding_cost",
    "verdict_from_proto",
    "verdict_from_wire",
    "verdict_schema_content_hash",
    "verdict_to_proto",
    "verdict_to_wire",
    "verdicts_to_table",
    "violations_per_scenario",
    "write_metrics_parquet",
]
