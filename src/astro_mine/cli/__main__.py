"""``python -m astro_mine.cli`` — parity with the component CLIs.

Every Astro-Mine CLI is runnable both as its console script and as ``python -m``; the module form
is what a container entrypoint and a ``uv run`` invocation reach for when the script directory is
not on ``PATH``.
"""

from __future__ import annotations

import sys

from astro_mine.cli._dispatch import main

if __name__ == "__main__":
    sys.exit(main())
