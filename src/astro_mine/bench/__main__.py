"""``python -m astro_mine.bench`` — dispatch to the Bench command line (RM-P0-BENCH-05)."""

from __future__ import annotations

from astro_mine.bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
