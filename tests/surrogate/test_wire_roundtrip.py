"""ErrorReport Protobuf wire form — exact, byte-stable round-trip (RM-P1-SURR-01).

The canonical cross-language encoding Sim's scheduler consumes (surrogate.md §5). A
model -> bytes -> model round-trip must be exact, and the bytes must be deterministic, so a
content hash over the wire form is portable (the Core messages.wire pattern).
"""

from __future__ import annotations

import pytest

from astro_mine.surrogate import ErrorReport
from astro_mine.surrogate.wire import (
    error_report_from_wire,
    error_report_to_proto,
    error_report_to_wire,
)
from tests.surrogate.factories import granular_report, illumination_report


@pytest.mark.parametrize(
    "report", [granular_report(), illumination_report()], ids=["granular", "illumination"]
)
def test_round_trip_is_exact(report: ErrorReport) -> None:
    assert error_report_from_wire(error_report_to_wire(report)) == report


@pytest.mark.parametrize(
    "report", [granular_report(), illumination_report()], ids=["granular", "illumination"]
)
def test_serialization_is_byte_stable(report: ErrorReport) -> None:
    assert error_report_to_wire(report) == error_report_to_wire(report)


def test_optional_rollout_absence_round_trips() -> None:
    # A field surrogate omits the rollout facet; absence must survive the wire.
    report = illumination_report()
    assert report.rollout is None
    assert error_report_from_wire(error_report_to_wire(report)).rollout is None


def test_rollout_presence_round_trips() -> None:
    report = granular_report()
    assert report.rollout is not None
    restored = error_report_from_wire(error_report_to_wire(report))
    assert restored.rollout == report.rollout


def test_to_proto_exposes_the_typed_message() -> None:
    msg = error_report_to_proto(granular_report())
    assert msg.surrogate_name == "excavation-gnn"
    assert msg.domain == "granular_excavation"
