"""Decode a Sim MCAP recording into a Bench :class:`EpisodeTrace` (RM-P0-BENCH-07; bench.md §6).

The Bench-side consumer of Sim's output at the **MCAP artifact boundary** ("Sim rollouts produce
MCAP → metrics compute", bench.md §6): read the Core ``Observation`` stream a Sim run recorded and
assemble the :class:`~astro_mine.bench.metrics.EpisodeTrace` the metric
set scores — so Bench scores a **real Sim run without importing** ``astro_mine.sim``. The ``mcap``
library is a *data-format reader*, not a Sim import; it and this subpackage require the
``[recording]`` extra, so the base package stays dependency-clean (core + pydantic) — ``mcap`` is
imported lazily, so ``import astro_mine.bench.recording`` works without the extra until decode time.

Sim's recording format ([astro-mine-sim RM-P0-SIM-09](https://github.com/astro-mine/astro-mine-sim))
is the contract this reads:

- per-tick canonical frames on the MCAP topic ``/sim/frames`` — JSON objects
  ``{"kind": "reset"|"step", "observations": {agent_id: <Observation json>}, ...}``;
- a ``provenance.json`` attachment ``{"content_hash", "run": {"seed", ...}, "environment"}``.

The scorer-owned :class:`~astro_mine.bench.metrics.ScoringContext` (belief/PSR/night inputs,
thresholds, species — prospect.md §9) is **not** in the recording; the caller (the scenario)
supplies it, defaulting to an empty context.

Backlog: RM-P0-BENCH-07 — https://github.com/astro-mine/astro-mine-bench/issues/13
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from astro_mine.core.messages import Observation
from astro_mine.core.scoring import EpisodeTrace, ScoringContext

__all__ = [
    "FRAMES_TOPIC",
    "PROVENANCE_ATTACHMENT",
    "DecodedRecording",
    "RecordingError",
    "decode_recording",
]

#: The MCAP topic Sim writes its per-tick canonical frames to (astro-mine-sim RM-P0-SIM-09).
FRAMES_TOPIC = "/sim/frames"
#: The MCAP attachment carrying Sim's JSON provenance envelope.
PROVENANCE_ATTACHMENT = "provenance.json"


class RecordingError(ValueError):
    """Raised when an MCAP is not a scorable Sim recording (no ``/sim/frames`` channel)."""


@dataclass(frozen=True, slots=True)
class DecodedRecording:
    """A Sim MCAP decoded for scoring: the provenance ``seed`` + ``content_hash`` and the trace.

    ``seed`` and ``content_hash`` come from the recording's ``provenance.json`` (``None``/empty when
    absent); ``trace`` is the :class:`EpisodeTrace` the metric set scores. A caller scores many
    seeds by decoding each recording and mapping ``seed → trace`` into
    :func:`~astro_mine.bench.metrics.score`.
    """

    seed: int | None
    content_hash: str
    trace: EpisodeTrace


def decode_recording(
    path: str | Path, *, context: ScoringContext | None = None
) -> DecodedRecording:
    """Decode a Sim MCAP at ``path`` into a scorable :class:`DecodedRecording`.

    Reads every ``/sim/frames`` message (reset + steps) and flattens each frame's per-agent
    :class:`~astro_mine.core.messages.Observation` into the trace's observation tuple, in log order;
    ``context`` is the scorer-owned :class:`ScoringContext` (default: empty). Reads the provenance
    attachment for the ``seed`` and ``content_hash``. Raises :class:`RecordingError` if the file has
    no ``/sim/frames`` channel (not a Sim recording); ``astro_mine.sim`` is never imported.
    """
    from mcap.reader import make_reader  # lazily: the [recording] extra supplies mcap

    observations: list[Observation] = []
    envelope: dict[str, object] = {}
    saw_frames_channel = False
    with Path(path).open("rb") as handle:
        reader = make_reader(handle)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic != FRAMES_TOPIC:
                continue
            saw_frames_channel = True
            frame = json.loads(message.data)
            for observation in frame.get("observations", {}).values():
                observations.append(Observation.model_validate(observation))
        for attachment in reader.iter_attachments():
            if attachment.name == PROVENANCE_ATTACHMENT:
                envelope = json.loads(attachment.data)
                break
    if not saw_frames_channel:
        raise RecordingError(f"{path}: no {FRAMES_TOPIC!r} channel — not a Sim recording")

    run = envelope.get("run", {})
    seed = run.get("seed") if isinstance(run, dict) else None
    trace = EpisodeTrace(observations=tuple(observations), context=context or ScoringContext())
    return DecodedRecording(
        seed=seed if isinstance(seed, int) else None,
        content_hash=str(envelope.get("content_hash", "")),
        trace=trace,
    )
