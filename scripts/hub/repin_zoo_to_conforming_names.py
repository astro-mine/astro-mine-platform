#!/usr/bin/env python3
"""Re-pin every zoo scenario onto the conforming artifact names (``conventions.md`` §13).

The second half of the one-sweep migration. :mod:`scripts.hub.migrate_artifact_names` re-publishes
the content and writes an ``old name -> {new_name, manifest_digest}`` map; this applies that map to
the scenarios, so the two halves cannot disagree about which digest a name got.

**What migrates and what must not.** §13 draws the line at whether a reference is *live*. A zoo
``pins.json`` and a ``scenario.json`` say what the task resolves against today, so they migrate. A
``PROVENANCE.md``, a ``RECIPE.md`` or a published scorecard says what *was* published, under the
name it was published under, beside its digest — rewriting those makes the record false, and this
script does not touch them.

Each affected scenario needs a new ``spec_version``: a content pin is bound into the spec hash, so
re-pinning in place would silently invalidate every leaderboard result pinned to the old spec
(``bench.md`` §5, §8). Every scenario carries its own version line, so the minor is bumped
per scenario rather than set to one shared value. Results scored against the old version stay valid
for that version.

``--already-bumped`` names a scenario that was *already* moved to its new version earlier in the
same change, and must not be bumped twice. The anchor is the case this exists for: rotating the
held-out seeds took it to ``0.9.0``, and the whole point of doing the rotation and the rename
together is that the zoo is superseded **once** rather than twice.

    python scripts/hub/repin_zoo_to_conforming_names.py \\
        --digests migration-digests.json --already-bumped lunar-polar-ice-prospecting-v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ZOO = REPO / "src" / "astro_mine" / "bench" / "zoo"


def _load_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """``old name -> record``, merged across every digest map given."""
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for old, record in json.loads(path.read_text(encoding="utf-8")).items():
            if record.get("dry_run"):
                raise SystemExit(f"{path} is a dry-run map; it carries no digests")
            merged[old] = record
    return merged


def _repin_pins(path: Path, mapping: dict[str, dict[str, Any]]) -> int:
    """Rewrite a ``pins.json``: the key, ``content_id``, the version, and names in the recipe."""
    pins: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    changed = 0
    for name, entry in pins.items():
        record = mapping.get(name)
        if record is None:
            out[name] = entry
            continue
        new = str(record["new_name"])
        entry = dict(entry)
        entry["content_id"] = new
        # The descriptor's version drifted from the spec's actual pin for the ISRU plant: 0.8.0
        # re-pinned it to 0.2.0 and recorded that in PROVENANCE.md and scenario.json, while this
        # file still said 0.1.0. The published artifact is the authority, so take the version from
        # the map rather than preserving the descriptor's claim.
        entry["content_version"] = str(record["version"])
        recipe = str(entry.get("recipe", ""))
        # Only the artifact's own name is rewritten. Cross-references to PROVENANCE.md and the
        # issue history stay as written — they are the historical record §13 says not to touch.
        recipe = re.sub(rf"\b{re.escape(name)}\b", new, recipe)
        entry["recipe"] = recipe
        out[new] = entry
        changed += 1
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def _bump_minor(version: str) -> str:
    major, minor, _patch = [*version.split("."), "0", "0"][:3]
    return f"{major}.{int(minor) + 1}.0"


def _repin_scenario(path: Path, mapping: dict[str, dict[str, Any]], bump: bool) -> int:
    spec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    changed = 0

    def apply(ref: dict[str, Any]) -> None:
        nonlocal changed
        record = mapping.get(str(ref.get("id")))
        if record is None:
            return
        ref["id"] = str(record["new_name"])
        ref["content_hash"] = str(record["manifest_digest"])
        changed += 1

    content = spec.get("content", {})
    if isinstance(content.get("world"), dict):
        apply(content["world"])
    for key in ("fleet", "prospect"):
        for ref in content.get(key) or []:
            apply(ref)
    if isinstance(content.get("link"), dict):
        apply(content["link"])

    # `placement.sites[].asset` names a pinned fleet asset, and it is a *live* reference like any
    # other -- `_spec.py` validates that placement keys the same vocabulary the content pins, so
    # missing it here does not drift silently, it refuses to load. Named explicitly rather than
    # swept, because a blanket walk would also rewrite the historical strings §13 protects.
    for site in spec.get("placement", {}).get("sites") or []:
        record = mapping.get(str(site.get("asset")))
        if record is not None:
            site["asset"] = str(record["new_name"])
            changed += 1

    if changed and bump:
        spec["spec_version"] = _bump_minor(str(spec["spec_version"]))
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--digests", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--already-bumped",
        nargs="*",
        default=(),
        help="Scenario ids already moved to their new spec_version earlier in this change.",
    )
    args = parser.parse_args(argv)

    mapping = _load_map(args.digests)
    print(f"{len(mapping)} migrated name(s) in the map")

    total = 0
    already = set(args.already_bumped)
    for scenario_dir in sorted(p for p in ZOO.iterdir() if (p / "scenario.json").is_file()):
        spec = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
        bump = str(spec["scenario_id"]) not in already
        pins = _repin_pins(scenario_dir / "pins.json", mapping)
        refs = _repin_scenario(scenario_dir / "scenario.json", mapping, bump)
        after = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
        total += refs
        note = "" if bump else "  (already bumped)"
        print(
            f"  {scenario_dir.name}: {pins} pin(s), {refs} ref(s), "
            f"spec_version {spec['spec_version']} -> {after['spec_version']}{note}"
        )
    print(f"\n{total} content reference(s) re-pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
