"""Fail-safe rejection branches of the loader (safety-critical; guard.md §2, §9.1).

A safety contract must reject anything it cannot certify. These cases exercise every
structural/semantic guard so an ill-formed or unbounded spec fails loudly at authoring time
rather than being silently admitted.
"""

from __future__ import annotations

import pytest

from astro_mine.guard.spec import SafetySpecValidationError, load_safety_spec, validate_safety_spec

_SIGNALS = "[{key: s, unit: J, source: observation}, {key: t, unit: K, source: observation}]"


def _load(constraints: str, signals: str = _SIGNALS) -> None:
    load_safety_spec(
        f"""
safety_version: "0.1"
safety:
  id: t
  name: t
  signals: {signals}
  constraints:
{constraints}
"""
    )


def _temporal(formula: str) -> str:
    return f"""
    - kind: temporal
      id: tc
      temporal:
        formula:
{formula}
"""


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        # unbounded / ill-formed intervals — the core fail-safe invariant
        (
            "          op: always\n"
            "          interval_s: {lo: 10.0, hi: 5.0}\n"
            "          args: [{op: predicate, signal: s, cmp: ge, threshold: 1.0}]",
            "well-ordered",
        ),
        (
            "          op: eventually\n"
            "          interval_s: {lo: 0.0, hi: .inf}\n"
            "          args: [{op: predicate, signal: s, cmp: ge, threshold: 1.0}]",
            "must be finite",
        ),
        # arity violations
        (
            "          op: not\n"
            "          args:\n"
            "            - {op: predicate, signal: s, cmp: ge, threshold: 1.0}\n"
            "            - {op: predicate, signal: t, cmp: le, threshold: 9.0}",
            "exactly one operand",
        ),
        (
            "          op: and\n"
            "          args: [{op: predicate, signal: s, cmp: ge, threshold: 1.0}]",
            "at least two operands",
        ),
        (
            "          op: always\n"
            "          interval_s: {lo: 0.0, hi: 5.0}\n"
            "          args:\n"
            "            - {op: predicate, signal: s, cmp: ge, threshold: 1.0}\n"
            "            - {op: predicate, signal: t, cmp: le, threshold: 9.0}",
            "exactly one operand",
        ),
        (
            "          op: until\n"
            "          interval_s: {lo: 0.0, hi: 5.0}\n"
            "          args: [{op: predicate, signal: s, cmp: ge, threshold: 1.0}]",
            "exactly two operands",
        ),
        # predicate/operand field misuse
        (
            "          op: predicate\n"
            "          signal: s\n"
            "          cmp: ge\n"
            "          threshold: 1.0\n"
            "          args: [{op: predicate, signal: t, cmp: le, threshold: 9.0}]",
            "takes no operands",
        ),
        (
            "          op: predicate\n          signal: s\n          cmp: ge",
            "requires signal, cmp, and threshold",
        ),
        (
            "          op: and\n"
            "          signal: s\n"
            "          args:\n"
            "            - {op: predicate, signal: s, cmp: ge, threshold: 1.0}\n"
            "            - {op: predicate, signal: t, cmp: le, threshold: 9.0}",
            "must not set predicate fields",
        ),
        (
            "          op: always\n"
            "          args: [{op: predicate, signal: s, cmp: ge, threshold: 1.0}]",
            "requires a bounded interval_s",
        ),
        (
            "          op: predicate\n"
            "          signal: s\n"
            "          cmp: ge\n"
            "          threshold: 1.0\n"
            "          interval_s: {lo: 0.0, hi: 1.0}",
            "takes no interval",
        ),
        (
            "          op: not\n"
            "          interval_s: {lo: 0.0, hi: 1.0}\n"
            "          args: [{op: predicate, signal: s, cmp: ge, threshold: 1.0}]",
            "takes no interval",
        ),
        # undeclared predicate signal
        (
            "          op: always\n"
            "          interval_s: {lo: 0.0, hi: 5.0}\n"
            "          args: [{op: predicate, signal: ghost, cmp: ge, threshold: 1.0}]",
            "not declared in signals",
        ),
    ],
)
def test_formula_rejections(formula: str, match: str) -> None:
    with pytest.raises(SafetySpecValidationError, match=match):
        _load(_temporal(formula))


def test_reject_two_union_payloads_set() -> None:
    # kind says power_floor but both power_floor and energy_floor payloads are set.
    with pytest.raises(SafetySpecValidationError, match="also sets"):
        _load(
            """
    - kind: power_floor
      id: p
      power_floor: {signal: s, floor_w: 1.0}
      energy_floor: {signal: s, floor_j: 2.0}
"""
        )


def test_reject_keep_out_shape_mismatch() -> None:
    # shape says box but a sphere payload is set instead.
    with pytest.raises(SafetySpecValidationError, match="box"):
        _load(
            """
    - kind: keep_out
      id: ko
      keep_out:
        margin_m: 1.0
        volume:
          shape: box
          sphere: {frame: F, center_m: {x: 0, y: 0, z: 0}, radius_m: 1.0}
"""
        )


def test_reject_duplicate_signal_keys() -> None:
    with pytest.raises(SafetySpecValidationError, match="duplicate signal key"):
        _load(
            """
    - {kind: energy_floor, id: e, energy_floor: {signal: s, floor_j: 1.0}}
""",
            signals=(
                "[{key: s, unit: J, source: observation}, {key: s, unit: K, source: observation}]"
            ),
        )


def test_reject_empty_signal_key() -> None:
    with pytest.raises(SafetySpecValidationError, match="non-empty"):
        _load(
            """
    - {kind: energy_floor, id: e, energy_floor: {signal: '', floor_j: 1.0}}
""",
            signals="[{key: '', unit: J, source: observation}]",
        )


def test_validate_text_path() -> None:
    # validate_safety_spec on raw text goes through the full loader (structural + semantic).
    with pytest.raises(SafetySpecValidationError):
        validate_safety_spec("safety_version: '0.1'")  # missing safety


def test_validate_dict_path_rejects_bad_structure() -> None:
    with pytest.raises(SafetySpecValidationError):
        validate_safety_spec({"safety_version": "0.1"})  # missing safety
