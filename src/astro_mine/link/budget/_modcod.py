"""The modulation/coding table — CCSDS-aligned required Eb/N0 (LINK-03).

A :class:`ModCod` names a waveform (modulation + code rate) and its **required Eb/N0** — the
energy-per-bit-to-noise-density a receiver needs to close at the target error rate. The link
budget converts available C/N0 into an achievable bit rate against this threshold.

The default :data:`CCSDS_MODCODS` carries representative values aligned with CCSDS 131.0-B
(TM synchronization & channel coding) at a reference FER ≈ 1e-4. They are *representative for a
parametric budget*, not certified link-acceptance numbers — and the table is **data, not code**
(link.md §3, "mod/cod tables are data"): pass a different :class:`ModCodTable` to model another
coding family. Rigorous regression against worked CCSDS examples is LINK-05.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from astro_mine.link.budget._errors import LinkBudgetError, ModCodError

__all__ = ["CCSDS_MODCODS", "ModCod", "ModCodTable"]


@dataclass(frozen=True)
class ModCod:
    """A modulation/coding scheme: its required Eb/N0 and spectral efficiency.

    ``required_ebn0_db`` is the Eb/N0 (dB) to close at the reference error rate;
    ``spectral_efficiency_bps_per_hz`` is the information bits per symbol times the code rate
    (informational — the Phase-0 budget is energy-limited, not bandwidth-limited).
    """

    name: str
    required_ebn0_db: float
    spectral_efficiency_bps_per_hz: float


class ModCodTable:
    """An immutable, name-indexed set of :class:`ModCod`s — swappable budget data.

    Construct from any sequence of schemes; duplicate names are rejected at construction so a
    table is unambiguous.
    """

    def __init__(self, schemes: Sequence[ModCod]) -> None:
        by_name: dict[str, ModCod] = {}
        for scheme in schemes:
            if scheme.name in by_name:
                raise LinkBudgetError(f"duplicate mod/cod {scheme.name!r} in table")
            by_name[scheme.name] = scheme
        self._by_name = by_name

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    @property
    def names(self) -> frozenset[str]:
        """The mod/cod names this table defines."""
        return frozenset(self._by_name)

    def get(self, name: str) -> ModCod:
        """The :class:`ModCod` named ``name``, or raise :class:`ModCodError` if unknown."""
        try:
            return self._by_name[name]
        except KeyError:
            raise ModCodError(
                f"unknown mod/cod {name!r}; table defines {sorted(self._by_name)}"
            ) from None


#: Representative CCSDS-aligned mod/cods (FER ≈ 1e-4). Required Eb/N0 rises with code rate and
#: modulation order; BPSK/QPSK share an Eb/N0 threshold (QPSK = two orthogonal BPSK), GMSK
#: carries a small filtering penalty. Covers the waveforms the anchor fleet's radios declare.
CCSDS_MODCODS = ModCodTable(
    [
        ModCod("bpsk_r1_2", required_ebn0_db=1.0, spectral_efficiency_bps_per_hz=0.5),
        ModCod("qpsk_r1_2", required_ebn0_db=1.0, spectral_efficiency_bps_per_hz=1.0),
        ModCod("qpsk_r3_4", required_ebn0_db=2.8, spectral_efficiency_bps_per_hz=1.5),
        ModCod("gmsk_r1_2", required_ebn0_db=1.5, spectral_efficiency_bps_per_hz=0.5),
        ModCod("gmsk_r3_4", required_ebn0_db=3.3, spectral_efficiency_bps_per_hz=0.75),
        ModCod("8psk_r3_4", required_ebn0_db=5.4, spectral_efficiency_bps_per_hz=2.25),
    ]
)
