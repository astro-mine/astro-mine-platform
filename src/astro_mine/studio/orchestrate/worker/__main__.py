"""``python -m astro_mine.studio.orchestrate.worker`` — the argv Cloud runs.

Deliberately separate from the package body: :mod:`astro_mine.studio.orchestrate.worker` is
imported by :mod:`~astro_mine.studio.orchestrate.cloud` (which needs the request/outcome models
to build and read a job), so running the package module itself with ``-m`` would execute it a
second time under the name ``__main__`` — the ``runpy`` double-import warning. A ``__main__``
submodule is never imported by anything, so it executes exactly once.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
