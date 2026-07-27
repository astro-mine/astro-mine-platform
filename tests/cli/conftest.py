"""Shared test setup.

The fast lane builds environments by injecting :class:`~importlib.metadata.EntryPoint` objects
rather than installing packages, and those entry points resolve against :mod:`_verbs`. Pytest's
default import mode already puts this directory on ``sys.path``; doing it explicitly means the
suite does not depend on that behaviour surviving a config change.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
