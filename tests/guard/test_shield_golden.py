"""PolicyShield determinism golden gate (RM-P1-GUARD-03/-06; guard.md §9.3, §10).

The seeded reproducibility contract: a fixed observation sequence with ``watchdog=False`` produces a
fixed sequence of certified actions and verdict provenance, pinned byte-for-byte in
``tests/golden/shield_run.json`` (regenerate with ``scripts/gen_shield_golden.py``). CI fails on any
non-reproducibility — the certify/correct/fall-back behaviour and the verdict provenance must be
identical across runs and machines, so a shielded run's safety behaviour is auditable/replayable.
"""

from __future__ import annotations

import json
from pathlib import Path

from astro_mine.core.hashing import canonical_json, content_hash_json
from tests.guard.shield_fixture import run_shield

GOLDEN = Path(__file__).resolve().parent / "golden" / "shield_run.json"


def test_matches_checked_in_golden() -> None:
    provenance = [v.provenance() for v in run_shield()]
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert provenance == golden["run"]
    assert content_hash_json(provenance) == golden["digest"]


def test_double_run_is_byte_identical() -> None:
    # Determinism, not just golden-match: two independent runs canonicalize to identical bytes.
    first = [v.provenance() for v in run_shield()]
    second = [v.provenance() for v in run_shield()]
    assert canonical_json(first) == canonical_json(second)


def test_golden_exercises_all_three_layers() -> None:
    # The reproducibility artifact must cover primary certify, shield correct, and backup fall-back.
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert [v["layer"] for v in golden["run"]] == ["primary", "shield", "backup"]
    assert golden["run"][2]["constraint_ids"] == ["c_anchor_torque"]
