"""View data + MCAP replay export — Bench's side of the View handoff (RM-P1-BENCH-12; bench.md §6).

bench.md §6: "[View] surfaces leaderboards, scorecards, and replays of evaluation episodes; **Bench
provides the data and MCAP replays, View renders them**." View is a separate component (formally
Phase 2; no repo yet) and a *co-dependency* of both this issue and Studio's embedded-View work
(RM-P1-STUDIO-06). This module is Bench's half of that contract — the data shapes and replay
packaging View consumes — decoupled from any View code:

- :func:`export_leaderboard` → :class:`ViewLeaderboard`: the **full per-metric** leaderboard data.
  Each row carries *every* scored metric with its value, direction, and cross-seed **uncertainty**
  (``dispersion``), unlike the primary-metric-only :class:`LeaderboardEntry`, so View renders
  scorecards and surfaces bounds rather than a single number (LUNAR-UX-006).
- :func:`replay_manifest` → :class:`ViewReplay`: decode a Sim MCAP episode recording into a
  lightweight replay manifest (frame/observation counts, agents, sim-time span, content digest); the
  MCAP bytes themselves are the replay payload View plays.

Requires the ``[leaderboard]`` extra (the leaderboard models + ranking) and, for
:func:`replay_manifest`, the ``[recording]`` extra (the MCAP reader). Bench never imports Sim.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from astro_mine.bench.leaderboard._eval import rank
from astro_mine.bench.leaderboard._models import Integrity, MetricScore, Submission
from astro_mine.bench.leaderboard._objects import blob_digest
from astro_mine.bench.metrics import ScoringContext

__all__ = [
    "ViewLeaderboard",
    "ViewLeaderboardRow",
    "ViewReplay",
    "export_leaderboard",
    "replay_manifest",
]


class _Model(BaseModel):
    """Frozen, extra-forbidding base for the View-facing data shapes (Bench-owned, not Core)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ViewLeaderboardRow(_Model):
    """One ranked row carrying its **full** per-metric scorecard — the shape View renders.

    Unlike :class:`~astro_mine.bench.leaderboard.LeaderboardEntry` (primary metric only), ``scores``
    holds every scored metric with its ``value``, ``direction``, and ``dispersion`` (the cross-seed
    uncertainty View shows as a bound). ``trace_hash`` is the stored MCAP replay's digest (``None``
    when no replay is attached), so View knows whether an episode replay is available.

    ``runner`` is the identity of the runner that produced the scorecard (``"fixture/0.1.0"`` for
    the reference fixture, else a Sim runner's id). View **must** render it in the ranking row: a
    fixture-scored entry has to *look* fixture-scored, not merely carry a footnote (G1.1's lesson
    applied to pixels — gap report §8.2.6). Surfacing it here is what lets a leaderboard tell a
    simulated result from a fixture one by provenance rather than by value (G1.8).
    """

    rank: int
    submission_id: str
    method: str | None
    author: str | None
    integrity: Integrity
    runner: str
    source: str | None
    provenance_hash: str | None
    trace_hash: str | None
    scores: tuple[MetricScore, ...]


class ViewLeaderboard(_Model):
    """A scenario's leaderboard as View consumes it: the primary metric + full-metric rows."""

    scenario_id: str
    primary_metric: str | None
    rows: tuple[ViewLeaderboardRow, ...]


class ViewReplay(_Model):
    """A decoded replay manifest for a Sim MCAP episode — the metadata View needs to render it.

    ``mcap_digest``/``size_bytes`` address the replay payload (the MCAP bytes View plays);
    ``frame_count`` is the number of distinct sim ticks, ``observation_count`` the per-agent
    observations flattened across them; ``agents`` the distinct agents; ``sim_time_start_s`` /
    ``sim_time_end_s`` the episode's sim-time span. ``seed`` + ``content_hash`` come from the
    recording's provenance envelope, tying the replay back to its scored run.
    """

    scenario_id: str
    submission_id: str | None
    mcap_digest: str
    size_bytes: int
    frame_count: int
    observation_count: int
    agents: tuple[str, ...]
    sim_time_start_s: float | None
    sim_time_end_s: float | None
    seed: int | None
    content_hash: str


def export_leaderboard(scenario_id: str, submissions: Sequence[Submission]) -> ViewLeaderboard:
    """Export ``scenario_id``'s submissions as the full-metric :class:`ViewLeaderboard` for View.

    Rows are ordered by :func:`~astro_mine.bench.leaderboard.rank` (the same primary-metric ordering
    the ``/leaderboard`` edge uses), but each row carries its full per-metric scorecard — so View
    shows scorecards and uncertainty bounds, not just the ranked primary value.
    """
    entries = rank(list(submissions))
    by_id = {submission.submission_id: submission for submission in submissions}
    rows = tuple(
        ViewLeaderboardRow(
            rank=entry.rank,
            submission_id=entry.submission_id,
            method=by_id[entry.submission_id].method,
            author=by_id[entry.submission_id].author,
            integrity=by_id[entry.submission_id].integrity,
            runner=by_id[entry.submission_id].runner,
            source=by_id[entry.submission_id].source,
            provenance_hash=by_id[entry.submission_id].provenance_hash,
            trace_hash=by_id[entry.submission_id].trace_hash,
            scores=by_id[entry.submission_id].scores,
        )
        for entry in entries
    )
    primary_metric = entries[0].primary_metric if entries else None
    return ViewLeaderboard(scenario_id=scenario_id, primary_metric=primary_metric, rows=rows)


def replay_manifest(
    mcap: bytes,
    *,
    scenario_id: str,
    submission_id: str | None = None,
    context: ScoringContext | None = None,
) -> ViewReplay:
    """Decode a Sim MCAP episode recording (``mcap`` bytes) into a :class:`ViewReplay` manifest.

    Summarizes the recording — distinct ticks, per-agent observation count, agents, sim-time span,
    and provenance seed/hash — without importing Sim (the ``mcap`` reader is a data-format library).
    The bytes are addressed by their ``sha256:`` digest; the same bytes are the replay payload View
    plays. Raises :class:`~astro_mine.bench.recording.RecordingError` if ``mcap`` is not a Sim
    recording.
    """
    from astro_mine.bench.recording import decode_recording

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "replay.mcap"
        path.write_bytes(mcap)
        decoded = decode_recording(path, context=context)

    observations = decoded.trace.observations
    times = [obs.sim_time_s for obs in observations]
    return ViewReplay(
        scenario_id=scenario_id,
        submission_id=submission_id,
        mcap_digest=blob_digest(mcap),
        size_bytes=len(mcap),
        frame_count=len({obs.tick for obs in observations}),
        observation_count=len(observations),
        agents=tuple(sorted({obs.agent_id for obs in observations})),
        sim_time_start_s=min(times) if times else None,
        sim_time_end_s=max(times) if times else None,
        seed=decoded.seed,
        content_hash=decoded.content_hash,
    )
