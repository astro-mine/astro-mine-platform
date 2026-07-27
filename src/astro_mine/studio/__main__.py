"""``python -m astro_mine.studio`` — the same entry point as the ``astro-mine-studio`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
