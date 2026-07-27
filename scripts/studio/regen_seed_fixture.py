"""Regenerate the UI's seeded-example fixture from a real authoring run.

    uv run python scripts/regen_seed_fixture.py

The Studio surface opens on a seeded example so the workspace is **never empty on first open**
(`seed.py`). The browser renders a *committed* copy of that study, because the surface must open on
it before any backend answers — and a committed copy of a Pydantic model is a copy that can drift
from it.

It did. `TradeStudy.evaluator` became required in `9631069` and the fixture, last regenerated three
days earlier, had no such key: every open of every instance POSTed a `TradeStudy` that could not
validate, and the workspace showed *"comparison failed: evaluator: Field required"* where the Pareto
front belongs (issue #49).

So the fixture is **generated, not hand-edited**, and it is generated from the same
`_author_example()` the seeded campaign is built from — one source for the pinned artifact and the
rendered one. `tests/test_seed_fixture.py` round-trips the committed files through the real
`POST /studies/comparison`, so the next required field fails a test instead of a first open.

Volatile provenance (the toolchain and lockfile of whoever regenerates) is normalised, so re-running
this on another machine produces no diff unless the *models* changed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from astro_mine.studio.seed import example_study

UI_SRC = Path(__file__).resolve().parent.parent / "ui" / "src"
STUDY_PATH = UI_SRC / "seedStudy.json"
OBJECTIVE_PATH = UI_SRC / "seedObjective.json"

#: Provenance fields whose values are properties of the machine that regenerated the fixture, not of
#: the study. Pinned to a fixed value so a regeneration is a no-op diff unless a model changed.
_NORMALISED_PROVENANCE = {
    "code_version": "0.1.0",
    "toolchain_version": "python3.12",
    "env_lockfile": "sha256:" + "0" * 64,
}


def _normalise(value: Any) -> Any:
    """Recursively pin the environment-dependent provenance fields."""
    if isinstance(value, dict):
        pinned = {k: _normalise(v) for k, v in value.items()}
        for field, fixed in _NORMALISED_PROVENANCE.items():
            if field in pinned:
                pinned[field] = fixed
        return pinned
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    """The example objective document and trade study, as the JSON the UI imports."""
    objective, study = asyncio.run(example_study())
    return objective.model_dump(mode="json"), _normalise(study.model_dump(mode="json"))


def main() -> int:
    objective, study = build()
    OBJECTIVE_PATH.write_text(json.dumps(objective, indent=2) + "\n", encoding="utf-8")
    STUDY_PATH.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OBJECTIVE_PATH.relative_to(Path.cwd())}")
    print(f"wrote {STUDY_PATH.relative_to(Path.cwd())} (evaluator: {study['evaluator']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
