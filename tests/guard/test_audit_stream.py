"""The MCAP SafetyVerdict stream — replayable, best-effort telemetry (RM-P1-GUARD-06).

Proves the verdict stream round-trips through MCAP on the ``/guard/verdicts`` topic (so Bench / View
replay a shielded run channel-by-channel), survives a ``+inf`` margin, and — the safety MUST
— that a broken writer never raises into the caller (a dropped log never gates the shield;
guard.md §8, §9.1).
"""

from __future__ import annotations

import math
from pathlib import Path

from astro_mine.guard.audit import (
    VERDICTS_TOPIC,
    open_verdict_stream,
    read_verdicts,
)
from astro_mine.guard.audit.stream import VerdictStream
from tests.guard.conftest import make_verdict


def test_write_read_roundtrip(tmp_path: Path) -> None:
    verdicts = [make_verdict(tick=i, sim_time_s=float(i)) for i in range(3)]
    path = tmp_path / "verdicts.mcap"
    with open_verdict_stream(path) as stream:
        for v in verdicts:
            stream.write_verdict(v)
    back = read_verdicts(path)
    assert len(back) == 3
    assert [v.provenance() for v in back] == [v.provenance() for v in verdicts]


def test_infinity_margin_survives_the_stream(tmp_path: Path) -> None:
    verdict = make_verdict(
        min_barrier_margin=math.inf,
        layer="backup",
        intervention="fallback",
        reason="bad_input",
        backup_kind="brake_to_stop",
    )
    path = tmp_path / "inf.mcap"
    with open_verdict_stream(path) as stream:
        stream.write_verdict(verdict)
    (back,) = read_verdicts(path)
    assert math.isinf(back.min_barrier_margin)


def test_topic_and_schema_registered(tmp_path: Path) -> None:
    path = tmp_path / "topic.mcap"
    with open_verdict_stream(path) as stream:
        stream.write_verdict(make_verdict())
    from mcap.reader import make_reader

    topics = set()
    with path.open("rb") as handle:
        for _schema, channel, _message in make_reader(handle).iter_messages():
            topics.add(channel.topic)
    assert topics == {VERDICTS_TOPIC}


def test_empty_stream_is_well_formed(tmp_path: Path) -> None:
    path = tmp_path / "empty.mcap"
    with open_verdict_stream(path):
        pass
    assert read_verdicts(path) == []


def test_write_verdict_is_best_effort() -> None:
    # A writer that fails on add_message must not raise out of write_verdict.
    class BrokenWriter:
        def register_schema(self, **kwargs: object) -> int:
            return 1

        def register_channel(self, **kwargs: object) -> int:
            return 2

        def add_message(self, **kwargs: object) -> None:
            raise OSError("disk full")

    stream = VerdictStream(BrokenWriter())  # type: ignore[arg-type]
    stream.write_verdict(make_verdict())  # must not raise — telemetry is best-effort


def test_non_finite_sim_time_maps_to_zero_log_time() -> None:
    from astro_mine.guard.audit.stream import _log_time_ns

    assert _log_time_ns(2.5) == 2_500_000_000
    assert _log_time_ns(math.inf) == 0
    assert _log_time_ns(-1.0) == 0
