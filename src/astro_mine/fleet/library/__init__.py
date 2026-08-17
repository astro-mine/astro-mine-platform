"""Minimal reference asset library for the anchor scenario (RM-P0-FLEET-04).

The anchor "robot menu" in SADF: a relay orbiter, a lander, a prospecting rover
(neutron/NIR/GPR/drill), an excavator, a hauler, and a basic ISRU plant -- each with a
low-fidelity ``massmodel`` plus at least one higher-fidelity profile under one stable
identity (fleet.md §3/§12; scenario §6; LUNAR-FR-003).

Assets are authored as **SADF v0.1 YAML** documents under this package, laid out by the
``fleet.md`` library tree (``orbital/``, ``surface/``, ``manipulation/``, ``logistics/``,
``isru/``). Fleet *consumes* the waist: every document is parsed and validated through
Core's :func:`astro_mine.core.sadf.load_sadf`, so a reference asset cannot drift from the
schema. :func:`load_reference` returns the loaded :class:`~astro_mine.core.sadf.SadfDocument`
ready to lint (:func:`astro_mine.fleet.lint.lint_asset`), package
(:func:`astro_mine.fleet.packaging.package_asset`), and -- once RM-P0-FLEET-07 lands -- spawn
in Sim.

Backlog: RM-P0-FLEET-04 -- astro-mine-fleet#4
"""

from __future__ import annotations

from importlib import resources

from astro_mine.core.sadf import SadfDocument, load_sadf

__all__ = ["REFERENCE_ASSETS", "available", "load_reference"]

#: The anchor roster: stable reference name -> package-relative ``.sadf.yaml`` path. The
#: name is the human handle callers pass to :func:`load_reference`; the document's
#: ``identity.id`` (e.g. ``relay-orbiter``) is what Bench pins by content
#: hash. Append-only -- renaming a key breaks anyone who pinned the old handle.
REFERENCE_ASSETS: dict[str, str] = {
    "relay_orbiter": "orbital/relay-orbiter.sadf.yaml",
    "lander": "orbital/lander.sadf.yaml",
    "prospecting_rover": "surface/prospecting-rover.sadf.yaml",
    "excavator": "manipulation/excavator.sadf.yaml",
    "hauler": "logistics/hauler.sadf.yaml",
    "isru_plant": "isru/isru-plant.sadf.yaml",
}


def available() -> list[str]:
    """The reference asset names, sorted -- the menu :func:`load_reference` accepts."""
    return sorted(REFERENCE_ASSETS)


def load_reference(name: str) -> SadfDocument:
    """Load a named reference asset from the anchor-scenario library.

    ``name`` is one of :func:`available` (e.g. ``"prospecting_rover"``). The document is
    read from this package and validated through Core's loader, so the returned
    :class:`~astro_mine.core.sadf.SadfDocument` is guaranteed schema- and semantic-valid.

    Raises :class:`ValueError` for an unknown name (with the available menu), and
    :class:`~astro_mine.core.sadf.SadfValidationError` if a shipped asset fails validation
    (a packaging defect -- the reference library is covered by ``tests/test_library.py``).
    """
    try:
        rel_path = REFERENCE_ASSETS[name]
    except KeyError:
        raise ValueError(f"unknown reference asset {name!r}; available: {available()}") from None
    text = (resources.files(__name__) / rel_path).read_text(encoding="utf-8")
    return load_sadf(text)
