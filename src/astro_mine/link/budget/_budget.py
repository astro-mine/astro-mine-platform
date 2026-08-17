# SPDX-License-Identifier: Apache-2.0
"""The parametric link budget: gain / path loss / SNR → rate, plus latency (LINK-03).

Geometry is ground truth; RF is a layer on top (link.md §2.1). Given a transmit and receive
radio (SADF :class:`~astro_mine.core.sadf.model.Comms`) and a slant range from the geometry
layer (LINK-01), this computes free-space path loss, the carrier-to-noise-density C/N0, and the
achievable bit rate against a CCSDS-aligned mod/cod table — plus the one-way latency (light-time
+ turnaround). The Phase-0 budget is **energy-limited**: the rate a mod/cod supports is set by
how far C/N0 exceeds its required Eb/N0 (with margin), clamped to the radios' declared rate caps.

Reads the SADF ``Comms`` radio from ``astro_mine.core.sadf.model`` and ``CommsBand`` from
``astro_mine.core.sadf.enums`` — Core v0.1.0 keeps these in submodules (the package-level
re-export is a Core-side cleanup, not needed here).

**Degrade loudly.** Missing EIRP/G-T, a non-positive range/frequency, a band mismatch with no
explicit frequency, or radios sharing no mod/cod all raise :class:`LinkBudgetError`. A link that
closes on geometry but is too faint for any supported mod/cod is reported as ``feasible=False``
(rate 0) — a constraint a planner consumes, not an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from astro_mine.core.sadf.enums import CommsBand
from astro_mine.core.sadf.model import Comms
from astro_mine.link.budget._errors import LinkBudgetError
from astro_mine.link.budget._modcod import CCSDS_MODCODS, ModCodTable

__all__ = ["LinkBudget", "band_frequency_hz", "compute_link_budget"]

#: Speed of light in vacuum (m/s).
SPEED_OF_LIGHT_M_S = 299_792_458.0

#: Boltzmann's constant as 10·log10(k) in dBW/(K·Hz): the noise-density floor in C/N0.
BOLTZMANN_DBW_PER_K_HZ = -228.6

#: Representative center frequency per RF band (Hz). Overridable per call via ``frequency_hz``;
#: ``OPTICAL`` is intentionally absent — it is out of the parametric-RF budget (link.md §11).
_BAND_FREQUENCY_HZ: dict[CommsBand, float] = {
    CommsBand.UHF: 0.4e9,
    CommsBand.S_BAND: 2.2e9,
    CommsBand.X_BAND: 8.4e9,
    CommsBand.KA_BAND: 26.0e9,
}


def band_frequency_hz(band: CommsBand) -> float:
    """A representative center frequency (Hz) for ``band`` — or raise for an RF-less band."""
    try:
        return _BAND_FREQUENCY_HZ[band]
    except KeyError:
        raise LinkBudgetError(
            f"no representative frequency for band {band!r}; pass frequency_hz "
            "(optical is outside the parametric RF budget)"
        ) from None


@dataclass(frozen=True)
class LinkBudget:
    """The computed budget for one ``tx -> rx`` link at a given range.

    ``cn0_dbhz`` is the carrier-to-noise-density; ``rate_bps`` the achievable rate on the
    selected ``modcod`` (0 when ``feasible`` is ``False``); ``ebn0_db``/``margin_db`` are the
    energy-per-bit ratio and margin above the mod/cod's requirement at that rate (``None`` when
    infeasible). ``latency_s`` is one-way light-time plus a turnaround term.
    """

    range_m: float
    frequency_hz: float
    eirp_dbw: float
    fspl_db: float
    gt_db_per_k: float
    cn0_dbhz: float
    feasible: bool
    modcod: str | None
    rate_bps: float
    ebn0_db: float | None
    margin_db: float | None
    light_time_s: float
    latency_s: float


def _resolve_frequency(tx: Comms, rx: Comms, frequency_hz: float | None) -> float:
    """The link frequency: explicit ``frequency_hz``, else the shared band's center."""
    if frequency_hz is not None:
        if frequency_hz <= 0.0:
            raise LinkBudgetError(f"frequency_hz must be positive, got {frequency_hz}")
        return frequency_hz
    if tx.band != rx.band:
        raise LinkBudgetError(
            f"transmit band {tx.band!r} != receive band {rx.band!r}; pass frequency_hz"
        )
    return band_frequency_hz(tx.band)


def _eirp_dbw(tx: Comms) -> float:
    """The transmitter EIRP (dBW): its declared ``eirp_dbw``, else power + antenna gain."""
    if tx.eirp_dbw is not None:
        return tx.eirp_dbw
    gain = tx.antenna.gain_dbi if tx.antenna is not None else None
    if tx.tx_power_w is not None and tx.tx_power_w > 0.0 and gain is not None:
        return 10.0 * math.log10(tx.tx_power_w) + gain
    raise LinkBudgetError(
        f"transmit radio {tx.name!r} declares no EIRP and no tx_power_w + antenna gain to derive it"
    )


def _fspl_db(range_m: float, frequency_hz: float) -> float:
    """Free-space path loss (dB): 20·log10(4π·range·f / c)."""
    return 20.0 * math.log10(4.0 * math.pi * range_m * frequency_hz / SPEED_OF_LIGHT_M_S)


def _common_modcods(tx: Comms, rx: Comms) -> list[str]:
    """Mod/cods both radios support, in the transmitter's declared order (deduplicated)."""
    rx_supported = set(rx.modcod_supported)
    common = list(dict.fromkeys(m for m in tx.modcod_supported if m in rx_supported))
    if not common:
        raise LinkBudgetError(
            f"radios {tx.name!r} and {rx.name!r} share no mod/cod "
            f"({tx.modcod_supported} vs {rx.modcod_supported})"
        )
    return common


def _rate_cap_bps(tx: Comms, rx: Comms) -> float:
    """The lower of the two radios' max rates (∞ if neither declares one)."""
    caps = [r for r in (tx.max_rate_bps, rx.max_rate_bps) if r is not None]
    return min(caps) if caps else math.inf


def _rate_floor_bps(tx: Comms, rx: Comms) -> float:
    """The higher of the two radios' min rates (0 if neither declares one)."""
    floors = [r for r in (tx.min_rate_bps, rx.min_rate_bps) if r is not None]
    return max(floors) if floors else 0.0


def _select_modcod(
    cn0_dbhz: float,
    candidates: list[str],
    modcods: ModCodTable,
    *,
    margin_db: float,
    rate_cap_bps: float,
    rate_floor_bps: float,
) -> tuple[str, float, float, float] | None:
    """The supported mod/cod giving the highest feasible rate at ``cn0_dbhz``.

    Returns ``(name, rate_bps, ebn0_db, margin_db)`` or ``None`` if none clears the rate floor.
    Energy-limited: ``rate = C/N0 - required_Eb/N0 - margin`` (in dB-Hz -> bps), clamped to the
    radios' max rate; a clamped link keeps surplus Eb/N0 margin.
    """
    best: tuple[str, float, float, float] | None = None
    for name in candidates:
        scheme = modcods.get(name)
        rate_dbhz = cn0_dbhz - scheme.required_ebn0_db - margin_db
        rate_bps = min(10.0 ** (rate_dbhz / 10.0), rate_cap_bps)
        if rate_bps < rate_floor_bps:
            continue
        if best is None or rate_bps > best[1]:
            ebn0_db = cn0_dbhz - 10.0 * math.log10(rate_bps)
            best = (name, rate_bps, ebn0_db, ebn0_db - scheme.required_ebn0_db)
    return best


def compute_link_budget(
    tx: Comms,
    rx: Comms,
    *,
    range_m: float,
    frequency_hz: float | None = None,
    modcods: ModCodTable = CCSDS_MODCODS,
    margin_db: float = 3.0,
    extra_loss_db: float = 0.0,
    turnaround_s: float = 0.0,
) -> LinkBudget:
    """The parametric budget for a ``tx -> rx`` link at slant ``range_m``.

    Frequency defaults to the radios' shared band center (override with ``frequency_hz``);
    ``extra_loss_db`` lumps pointing/implementation/atmospheric losses; ``margin_db`` is the
    link margin reserved above each mod/cod's requirement; ``turnaround_s`` adds to the one-way
    light-time latency. Returns a :class:`LinkBudget`; raises :class:`LinkBudgetError` on
    missing/inconsistent inputs (see module docstring).
    """
    if range_m <= 0.0:
        raise LinkBudgetError(f"range_m must be positive, got {range_m}")

    frequency = _resolve_frequency(tx, rx, frequency_hz)
    eirp = _eirp_dbw(tx)
    if rx.gt_db_per_k is None:
        raise LinkBudgetError(f"receive radio {rx.name!r} declares no G/T (gt_db_per_k)")
    gt = rx.gt_db_per_k

    fspl = _fspl_db(range_m, frequency)
    cn0 = eirp - fspl + gt - BOLTZMANN_DBW_PER_K_HZ - extra_loss_db

    candidates = _common_modcods(tx, rx)
    best = _select_modcod(
        cn0,
        candidates,
        modcods,
        margin_db=margin_db,
        rate_cap_bps=_rate_cap_bps(tx, rx),
        rate_floor_bps=_rate_floor_bps(tx, rx),
    )

    light_time_s = range_m / SPEED_OF_LIGHT_M_S
    latency_s = light_time_s + turnaround_s

    if best is None:
        return LinkBudget(
            range_m=range_m,
            frequency_hz=frequency,
            eirp_dbw=eirp,
            fspl_db=fspl,
            gt_db_per_k=gt,
            cn0_dbhz=cn0,
            feasible=False,
            modcod=None,
            rate_bps=0.0,
            ebn0_db=None,
            margin_db=None,
            light_time_s=light_time_s,
            latency_s=latency_s,
        )
    name, rate_bps, ebn0_db, achieved_margin_db = best
    return LinkBudget(
        range_m=range_m,
        frequency_hz=frequency,
        eirp_dbw=eirp,
        fspl_db=fspl,
        gt_db_per_k=gt,
        cn0_dbhz=cn0,
        feasible=True,
        modcod=name,
        rate_bps=rate_bps,
        ebn0_db=ebn0_db,
        margin_db=achieved_margin_db,
        light_time_s=light_time_s,
        latency_s=latency_s,
    )
