"""``python -m astro_mine.hub.client`` → the CLI entry point."""

from __future__ import annotations

from astro_mine.hub.client.cli import main

if __name__ == "__main__":  # pragma: no cover - runtime shim
    raise SystemExit(main())
