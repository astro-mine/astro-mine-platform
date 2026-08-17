# SPDX-License-Identifier: Apache-2.0
"""``python -m astro_mine.bench eval-worker …`` — the single-seed rollout Cloud fans out.

**Not a command line.** Bench's user-facing verbs live in `astro-mine-cli` (`astro-mine bench
score`, `… fetch`, `… submit`, …). What remains here is one machine-facing entry point that
something already depends on: :func:`astro_mine.bench.eval.plan` builds this exact argv when it
fans an evaluation out per seed (RM-P1-BENCH-11), and the container backend runs the same argv
inside a sandbox (`TRUST_BOUNDARY.md`). It owns its own argparse and rides the
[cloud]/[recording] extras, which is why it was hidden from the old CLI's `--help` too.

Kept as `python -m` rather than a console script for the same reason the Cloud harness is: the
callers are images and job specs, and a module path is stable in a way a script name on `PATH`
is not (astro-mine-platform#1).
"""

from __future__ import annotations

import sys

_USAGE = (
    "python -m astro_mine.bench provides only `eval-worker` — the per-seed rollout Cloud fans "
    "out. Bench's command line is `astro-mine bench <verb>` (pip install astro-mine-cli)."
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "eval-worker":
        print(_USAGE, file=sys.stderr)
        return 2

    from astro_mine.bench.eval import run_worker

    return int(run_worker(args[1:]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
