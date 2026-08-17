# SPDX-License-Identifier: Apache-2.0
"""Guard's shipped reference safety content — resolvable from an installed wheel (GUARD, G2.7).

The reviewed anchor ``SafetySpec`` (lunar polar water-ice prospecting) is Guard's reference safety
contract. It ships here, **inside the package**, so it resolves from an installed wheel by name —
not from a path relative to the repo root, which only exists in a checkout. It previously lived at
``examples/safety_specs/anchor.safety.yaml``, a sibling of ``src/`` that maturin never packaged;
an installed Guard could not reach it, and ``astro-mine-mind`` had to *inline the whole document*
into its anchor stack to work around it — a second copy of a **safety** contract with nothing
keeping the two in sync. There is now one copy, and this is it.

Resolve it by name through :func:`anchor_safety_spec_text` / :func:`load_anchor_safety_spec`, never
by a filesystem path — ``importlib.resources`` reads it out of the installed distribution whether
that is a directory or a zip (the #55 / astro-mine-bench#37 wheel trap).
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.guard.spec.model import SafetyDocument

__all__ = [
    "ANCHOR_SAFETY_SPEC_RESOURCE",
    "anchor_safety_spec_text",
    "load_anchor_safety_spec",
]

#: The reviewed anchor SafetySpec, as a ``reference/``-relative resource name.
ANCHOR_SAFETY_SPEC_RESOURCE = "safety_specs/anchor.safety.yaml"


def anchor_safety_spec_text() -> str:
    """The reviewed anchor SafetySpec document, read from package data (offline, wheel-safe)."""
    return (
        resources.files("astro_mine.guard.reference")
        .joinpath(ANCHOR_SAFETY_SPEC_RESOURCE)
        .read_text(encoding="utf-8")
    )


def load_anchor_safety_spec() -> SafetyDocument:
    """Load and validate the reviewed anchor SafetySpec from package data.

    The one front door for the reference safety contract — consumers reference it **by name**, so
    there is never a second copy to drift. Imported lazily to keep this module free of the spec
    model at import time.
    """
    from astro_mine.guard.spec.loader import load_safety_spec

    return load_safety_spec(anchor_safety_spec_text())
