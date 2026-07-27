"""Parametric link budget — CCSDS-aligned (RM-P0-LINK-03).

Gain / path-loss / SNR → rate over a CCSDS-aligned mod/cod table, plus per-link latency
(light-time + turnaround). :func:`compute_link_budget` turns a transmit/receive SADF radio pair
and a slant range into a :class:`LinkBudget` (C/N0, achievable rate, Eb/N0 margin, latency);
:data:`CCSDS_MODCODS` is the default, swappable :class:`ModCodTable`. Geometry is ground truth;
RF is a layer on top (link.md §2.1). Degrades loudly on missing/inconsistent radio inputs.

Backlog: RM-P0-LINK-03 -- https://github.com/astro-mine/astro-mine-link/issues/3
"""

from __future__ import annotations

from astro_mine.link.budget._budget import (
    BOLTZMANN_DBW_PER_K_HZ,
    SPEED_OF_LIGHT_M_S,
    LinkBudget,
    band_frequency_hz,
    compute_link_budget,
)
from astro_mine.link.budget._errors import LinkBudgetError, ModCodError
from astro_mine.link.budget._modcod import CCSDS_MODCODS, ModCod, ModCodTable

__all__ = [
    "BOLTZMANN_DBW_PER_K_HZ",
    "CCSDS_MODCODS",
    "SPEED_OF_LIGHT_M_S",
    "LinkBudget",
    "LinkBudgetError",
    "ModCod",
    "ModCodError",
    "ModCodTable",
    "band_frequency_hz",
    "compute_link_budget",
]
