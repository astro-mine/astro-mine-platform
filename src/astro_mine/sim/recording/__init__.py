# SPDX-License-Identifier: Apache-2.0
"""Headless + interactive modes; MCAP recording + provenance stamping (RM-P0-SIM-09).

The output side of the stepping core: it records a run's per-tick frames to an **MCAP** container
(sim.md §3, §5) and stamps a **provenance envelope** sufficient to reproduce the run byte-for-byte —
every input content hash, every engine version/tier, the seed, and the scheduler's error-budget
outcomes (sim.md §5; conventions.md §5).

**Headless and interactive are one runtime** (sim.md §2.6). :func:`record_episode` drives the single
:func:`~astro_mine.sim.runtime.run_episode` loop through its ``on_frame`` sink, streaming each
canonical frame to the MCAP as it is produced; a headless run passes no extra sink, an interactive
run passes a live one (a View, a UI) — the same loop, different output sinks, never a divergent
codepath.

The envelope carries the same :attr:`~astro_mine.sim.runtime.Trace.content_hash` the RM-P0-SIM-10
determinism gate compares — there is **one** canonical artifact, not two. The deterministic part of
the envelope (seed, input hashes, engine versions, fidelity/error-budget outcomes) lives inside that
hash; the environment fingerprint (interpreter, package version) is recorded alongside but kept
*outside* the hash, so the determinism key stays portable across machines.

Backlog: RM-P0-SIM-09 -- astro-mine-sim#9
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcap.reader import make_reader
from mcap.writer import Writer

from astro_mine.sim.runtime.content import UnresolvedProvider
from astro_mine.sim.runtime.episode import Trace, run_episode

if TYPE_CHECKING:
    from astro_mine.core.policy import Policy
    from astro_mine.core.resource import ResourceField
    from astro_mine.core.world import WorldProvider
    from astro_mine.sim.comms import ConnectivitySource
    from astro_mine.sim.engines import EngineFactory
    from astro_mine.sim.runtime.scenario import Scenario
    from astro_mine.sim.runtime.timing import TimingRecorder

__all__ = [
    "FRAMES_TOPIC",
    "PROVENANCE_ATTACHMENT",
    "RunRecording",
    "open_recording",
    "read_recording",
    "record_episode",
    "run_provenance",
]

#: The MCAP topic the per-tick canonical frames are written to.
FRAMES_TOPIC = "/sim/frames"
#: The MCAP attachment name carrying the full JSON provenance envelope.
PROVENANCE_ATTACHMENT = "provenance.json"
#: A permissive JSON schema for the frame channel — the frames are heterogeneous reset/step records.
_FRAME_SCHEMA: dict[str, str] = {"type": "object", "title": "astro_mine.sim.Frame"}
_NS_PER_S = 1_000_000_000


def _sim_version() -> str:
    """The installed ``astro-mine-sim`` version (the env stamp's package fingerprint)."""
    return metadata.version("astro-mine-platform")


def _environment_fingerprint() -> dict[str, str]:
    """The recording-only environment stamp — sim package + interpreter + platform.

    Recorded for reproduction but deliberately kept *outside* the determinism hash (it is
    environment-dependent), so :attr:`Trace.content_hash` stays portable across machines."""
    return {
        "sim_version": _sim_version(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def run_provenance(trace: Trace, *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The full MCAP provenance envelope for ``trace`` — the byte-for-byte reproduction manifest.

    ``content_hash`` is the determinism key (RM-P0-SIM-10); ``run`` is the deterministic envelope
    (seed, scenario name, input content hashes, engine versions, and the scheduler's per-agent
    fidelity tier + error-budget outcomes); ``environment`` is the recording-only
    interpreter/package stamp (outside the hash) — the same ``content_hash`` the gate compares.

    ``timing`` is the run's measured wall-clock (:mod:`astro_mine.sim.runtime.timing`): the
    per-tier step cost and real-time factor sim.md §10 calls for. Like ``environment`` it sits
    *beside* the hash, never inside it — both are machine-dependent, and a determinism key that
    moved with either would fail the RM-P0-SIM-10 gate on every run and every host. Recorded so a
    replay can audit what the run cost; excluded so the key stays portable."""
    return {
        "content_hash": trace.content_hash,
        "run": dict(trace.provenance),
        "environment": dict(environment) if environment is not None else _environment_fingerprint(),
        "timing": dict(trace.timing) if trace.timing is not None else None,
    }


@dataclass(frozen=True, slots=True)
class RunRecording:
    """A recording read back from MCAP — frames, the provenance envelope, and the content hash."""

    content_hash: str
    provenance: Mapping[str, Any]
    frames: tuple[dict[str, Any], ...]


class _RecordingWriter:
    """Streams canonical frames to an open MCAP and stamps the provenance envelope.

    Obtained from :func:`open_recording`. :meth:`write_frame` is the per-tick sink the stepping
    loop drives (the ``on_frame`` hook); :meth:`write_provenance` stamps the envelope + content hash
    once the run completes."""

    def __init__(self, writer: Writer) -> None:
        self._writer = writer
        self._schema_id = writer.register_schema(
            name=_FRAME_SCHEMA["title"],
            encoding="jsonschema",
            data=json.dumps(_FRAME_SCHEMA, sort_keys=True).encode(),
        )
        self._channel_id = writer.register_channel(
            topic=FRAMES_TOPIC, message_encoding="json", schema_id=self._schema_id
        )
        self._sequence = 0

    def write_frame(self, frame: dict[str, Any]) -> None:
        """Write one canonical frame as a timestamped JSON message on the frames channel."""
        log_time = int(float(frame.get("sim_time_s", 0.0)) * _NS_PER_S)
        self._writer.add_message(
            channel_id=self._channel_id,
            log_time=log_time,
            data=json.dumps(frame, sort_keys=True).encode(),
            publish_time=log_time,
            sequence=self._sequence,
        )
        self._sequence += 1

    def write_provenance(
        self, trace: Trace, *, environment: Mapping[str, str] | None = None
    ) -> None:
        """Stamp the full provenance envelope (as a JSON attachment) plus headline metadata."""
        envelope = run_provenance(trace, environment=environment)
        self._writer.add_attachment(
            create_time=0,
            log_time=0,
            name=PROVENANCE_ATTACHMENT,
            media_type="application/json",
            data=json.dumps(envelope, sort_keys=True).encode(),
        )
        self._writer.add_metadata(
            "provenance",
            {
                "content_hash": trace.content_hash,
                "seed": str(trace.seed),
                "scenario": trace.scenario_name,
            },
        )


@contextmanager
def open_recording(path: str | Path) -> Iterator[_RecordingWriter]:
    """Open an MCAP recording at ``path`` for streaming frames + a provenance envelope.

    Yields a writer whose :meth:`~_RecordingWriter.write_frame` is the per-tick
    :func:`~astro_mine.sim.runtime.run_episode` ``on_frame`` sink and whose
    :meth:`~_RecordingWriter.write_provenance` stamps the envelope. The MCAP header and footer are
    written on entry and exit, so the file is well-formed even if nothing is recorded."""
    with Path(path).open("wb") as handle:
        writer = Writer(handle)
        writer.start(profile="astro-mine-sim", library=f"astro-mine-sim/{_sim_version()}")
        try:
            yield _RecordingWriter(writer)
        finally:
            writer.finish()


def record_episode(
    scenario: Scenario,
    path: str | Path,
    *,
    seed: int | None = None,
    resource_field: ResourceField | None = None,
    world_provider: WorldProvider | None = None,
    connectivity: ConnectivitySource | None = None,
    policy: Policy | None = None,
    engine_factory: EngineFactory | None = None,
    content_hashes: Mapping[str, str] | None = None,
    unresolved: Sequence[UnresolvedProvider] = (),
    on_frame: Callable[[dict[str, Any]], None] | None = None,
    environment: Mapping[str, str] | None = None,
    timing: TimingRecorder | None = None,
) -> Trace:
    """Run ``scenario`` and record it to an MCAP at ``path`` — the headless/interactive run modes.

    Streams each canonical frame to the MCAP as it is produced (one stepping loop), then stamps the
    provenance envelope. A **headless** run passes no ``on_frame``; an **interactive** run passes a
    live sink that sees each frame as it happens — the same loop, two sinks (sim.md §2.6).
    ``world_provider`` (the pinned world) and ``content_hashes`` (the ``id -> sha256:`` map a
    :class:`~astro_mine.sim.runtime.content.ContentResolver` returns) are threaded through too, so a
    **content-resolved** run records an MCAP that is byte-addressed to the exact bundles it ran on
    (RM-P1-SIM-01) — which is what makes a Bench score reproducible from the artifact alone.

    ``connectivity``, ``policy``, and ``engine_factory`` are threaded to
    :func:`~astro_mine.sim.runtime.run_episode` — comms masking (RM-P0-SIM-08), the injected Core
    decision loop, and the regime engine(s) (RM-P0-SIM-11) — so a recorded run is a Bench-scorable,
    policy-driven MCAP. Returns the canonical :class:`~astro_mine.sim.runtime.Trace`; its
    ``content_hash`` is also recorded in the file, so a re-run reproduces it byte-for-byte.

    ``timing`` is the wall-clock sink threaded to :func:`~astro_mine.sim.runtime.run_episode`; the
    measured cost is stamped into the envelope beside (never inside) the content hash."""

    with open_recording(path) as recording:

        def sink(frame: dict[str, Any]) -> None:
            recording.write_frame(frame)
            if on_frame is not None:
                on_frame(frame)

        trace = run_episode(
            scenario,
            seed=seed,
            resource_field=resource_field,
            world_provider=world_provider,
            connectivity=connectivity,
            policy=policy,
            engine_factory=engine_factory,
            content_hashes=content_hashes,
            unresolved=unresolved,
            on_frame=sink,
            timing=timing,
        )
        recording.write_provenance(trace, environment=environment)
    return trace


def read_recording(path: str | Path) -> RunRecording:
    """Read an MCAP recording back into frames + provenance + content hash (round-trip / Bench).

    The inverse of :func:`record_episode`: recovers every per-tick frame in log order and the full
    provenance envelope, so a consumer (Bench) reproduces and scores the run from the file alone.
    A recording with no provenance attachment yields an empty envelope and content hash."""
    frames: list[dict[str, Any]] = []
    envelope: dict[str, Any] = {}
    with Path(path).open("rb") as handle:
        reader = make_reader(handle)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic == FRAMES_TOPIC:
                frames.append(json.loads(message.data))
        for attachment in reader.iter_attachments():
            if attachment.name == PROVENANCE_ATTACHMENT:
                envelope = json.loads(attachment.data)
                break
    return RunRecording(
        content_hash=str(envelope.get("content_hash", "")),
        provenance=envelope,
        frames=tuple(frames),
    )
