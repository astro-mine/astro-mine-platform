"""Content-addressed curve aggregation — Parquet default, optional MLflow (RM-P1-LEARN-06).

The honest-eval curves are emitted as a tidy **long-format** table (one row per
``(algorithm, stress_axis, stress_value, seed)``) that Bench consumes and View replays
(learn.md §5, §10). Learn **defines** this versioned curve schema (``schema_version``, like
``CommsModelConfig``) as the cross-component contract; it is flagged for a later Bench
cross-check and adapted through the :class:`MetricSink` seam if Bench lands its own schema.

- :class:`CurveRow` / :data:`CURVE_SCHEMA_VERSION` — the versioned long-format row schema.
- :class:`CurveTable` — rows + the canonical **run manifest** whose
  :func:`~astro_mine.core.hashing.content_hash_json` is the reproducibility key
  (``manifest_hash``, stamped on every row); :meth:`~CurveTable.to_parquet` writes it via
  pyarrow (lazy — the lightweight ``[eval]`` extra).
- :class:`MetricSink` (Protocol) with the default :class:`ParquetSink` and a thin optional
  :class:`MlflowSink` (a lazy ``import mlflow`` behind the separate ``[mlflow]`` extra —
  MLflow is **not** a hard dependency). A full MLflow ``track/`` module (learn.md §3) is
  deferred; only this sink seam ships here.

pyarrow and mlflow are imported **inside** the methods that need them so importing this module
(and the whole ``eval`` surface) never requires either dependency (conventions.md §11).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from astro_mine.core.hashing import content_hash_json

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "CURVE_SCHEMA_VERSION",
    "CurveRow",
    "CurveTable",
    "MetricSink",
    "MlflowSink",
    "ParquetSink",
]

#: The long-format curve schema version — Learn owns this contract; bump on a shape change.
CURVE_SCHEMA_VERSION = "0.1.0"

#: The ordered column names of the long-format curve schema (the Bench/View contract).
CURVE_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "algorithm",
    "policy_id",
    "split",
    "stress_axis",
    "stress_value",
    "seed",
    "episode_return",
    "delivery_ratio",
    "offered",
    "delivered",
    "eval_throughput_steps_per_s",
    "wall_clock_s",
    "sample_efficiency",
    "comms_config_hash",
    "manifest_hash",
)


@dataclass(frozen=True)
class CurveRow:
    """One long-format curve observation: an ``(algorithm, stress point, seed)`` measurement.

    Carries the per-seed episode return and the comms-stress denominator
    (``delivery_ratio`` / ``offered`` / ``delivered``), the honest cost metrics, the
    content hash of the comms regime that produced it (``comms_config_hash``), and the
    ``policy_id`` (an ONNX package digest or ``live:<name>``)."""

    algorithm: str
    policy_id: str
    stress_axis: str
    stress_value: float
    seed: int
    episode_return: float
    delivery_ratio: float
    offered: int
    delivered: int
    eval_throughput_steps_per_s: float
    wall_clock_s: float
    sample_efficiency: float | None
    comms_config_hash: str
    split: str = "held_out"
    schema_version: str = CURVE_SCHEMA_VERSION


def _arrow_schema() -> pa.Schema:
    """The explicit pyarrow schema (stable column dtypes, nullable ``sample_efficiency``)."""
    import pyarrow as pa

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("algorithm", pa.string()),
            ("policy_id", pa.string()),
            ("split", pa.string()),
            ("stress_axis", pa.string()),
            ("stress_value", pa.float64()),
            ("seed", pa.int64()),
            ("episode_return", pa.float64()),
            ("delivery_ratio", pa.float64()),
            ("offered", pa.int64()),
            ("delivered", pa.int64()),
            ("eval_throughput_steps_per_s", pa.float64()),
            ("wall_clock_s", pa.float64()),
            ("sample_efficiency", pa.float64()),
            ("comms_config_hash", pa.string()),
            ("manifest_hash", pa.string()),
        ]
    )


class CurveTable:
    """A comms-stress curve as long-format rows plus its content-addressed run manifest.

    The ``manifest`` is the canonical description of the run (split, grid, seeds, Core
    interface versions, policy ids); its :func:`~astro_mine.core.hashing.content_hash_json`
    is the reproducibility key stamped on every row as ``manifest_hash``. Two runs with an
    identical manifest are the same experiment and — the determinism gate — must produce
    byte-identical rows."""

    def __init__(self, rows: Sequence[CurveRow], manifest: Mapping[str, Any]) -> None:
        self.rows: tuple[CurveRow, ...] = tuple(rows)
        self.manifest: dict[str, Any] = dict(manifest)
        #: The content address of the manifest (``sha256:…``) — the CX-REPRO key.
        self.manifest_hash: str = content_hash_json(self.manifest)

    def _record(self, row: CurveRow) -> dict[str, Any]:
        return {
            "schema_version": row.schema_version,
            "algorithm": row.algorithm,
            "policy_id": row.policy_id,
            "split": row.split,
            "stress_axis": row.stress_axis,
            "stress_value": float(row.stress_value),
            "seed": int(row.seed),
            "episode_return": float(row.episode_return),
            "delivery_ratio": float(row.delivery_ratio),
            "offered": int(row.offered),
            "delivered": int(row.delivered),
            "eval_throughput_steps_per_s": float(row.eval_throughput_steps_per_s),
            "wall_clock_s": float(row.wall_clock_s),
            "sample_efficiency": row.sample_efficiency,
            "comms_config_hash": row.comms_config_hash,
            "manifest_hash": self.manifest_hash,
        }

    def to_records(self) -> list[dict[str, Any]]:
        """The rows as plain dicts (each stamped with ``manifest_hash``) — the JSON/DataFrame
        view Bench and View consume without importing pyarrow."""
        return [self._record(row) for row in self.rows]

    def to_arrow(self) -> pa.Table:
        """The table as a pyarrow ``Table`` under the explicit curve schema (lazy pyarrow)."""
        import pyarrow as pa

        return pa.Table.from_pylist(self.to_records(), schema=_arrow_schema())

    def to_parquet(self, path: str | Path) -> Path:
        """Write the long-format curve to a Parquet file (the default aggregation sink)."""
        import pyarrow.parquet as pq

        out = Path(path)
        pq.write_table(self.to_arrow(), str(out))
        return out


@runtime_checkable
class MetricSink(Protocol):
    """The metric-aggregation seam: emit a scored :class:`CurveTable` somewhere.

    The default realization is :class:`ParquetSink`; :class:`MlflowSink` is an optional thin
    adapter. Bench can supply its own sink here without Learn taking an MLflow dependency."""

    def write(self, table: CurveTable) -> None: ...


class ParquetSink:
    """Write the curve to a content-addressed Parquet file — the default, dependency-light
    aggregation (the lightweight ``[eval]`` extra: pyarrow only)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def write(self, table: CurveTable) -> None:
        table.to_parquet(self._path)


class MlflowSink:
    """The optional MLflow :class:`MetricSink` — mirror a scored curve into MLflow.

    Now a thin **adapter over** :class:`~astro_mine.learn.track.backends.MlflowBackend`, so the
    package holds exactly **one** MLflow implementation: the full ``track/`` module (learn.md §3)
    ships, and this sink is its curve-only entry point — kept for callers that want to write a
    :class:`CurveTable` without holding a training run.

    MLflow is still **not** a hard dependency: the backend's ``import mlflow`` is lazy, so this
    fails loudly (``ImportError``) only if the sink is actually constructed without the
    ``[mlflow]`` extra installed. The backend import itself is *also* inside ``__init__`` — it
    both keeps this module MLflow-free at import and breaks the cycle (``track.run`` imports
    :class:`CurveTable` from here).

    Prefer :class:`~astro_mine.learn.track.TrackedRun` when you *do* have a training run: it
    lands the curves in the **same** MLflow run as the config, seeds, and provenance, rather than
    a standalone run that must later be joined back by hand."""

    def __init__(
        self,
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        experiment: str | None = None,
    ) -> None:
        from astro_mine.learn.track.backends import MlflowBackend

        self._backend = MlflowBackend(tracking_uri=tracking_uri, experiment=experiment)
        self._run_name = run_name

    def write(self, table: CurveTable) -> None:
        self._backend.start_run(self._run_name, {"manifest_hash": table.manifest_hash})
        try:
            self._backend.log_dict(table.manifest, "curve_manifest.json")
            for step, row in enumerate(table.rows):
                prefix = f"{row.algorithm}.{row.stress_axis}"
                self._backend.log_metrics(
                    {
                        f"{prefix}.episode_return": row.episode_return,
                        f"{prefix}.delivery_ratio": row.delivery_ratio,
                    },
                    step=step,
                )
        finally:
            self._backend.end_run()
