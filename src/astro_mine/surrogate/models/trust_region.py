# SPDX-License-Identifier: Apache-2.0
"""The excavation-parameter trust region + OOD flag (RM-P1-SURR-02).

Every surrogate declares the input domain it was trained/validated on; a query outside it raises
uncertainty and lowers the ``in_domain`` flag rather than a confident extrapolation
(surrogate.md principle 3 — "out-of-distribution silence is forbidden"). Here that domain is the
axis-aligned box over the excavation parameters the DEM fixture swept (density, friction,
restitution, tool speed). numpy only; no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from astro_mine.surrogate.report import Bound, TrustRegion

__all__ = ["ExcavationTrustRegion"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExcavationTrustRegion:
    """An axis-aligned box over the excavation parameters the surrogate was validated on."""

    param_names: tuple[str, ...]
    lower: FloatArray  # (P,)
    upper: FloatArray  # (P,)

    @classmethod
    def from_configs(
        cls, param_names: tuple[str, ...], configs: FloatArray
    ) -> ExcavationTrustRegion:
        """The tightest box enclosing the training configs ``(C, P)``."""
        return cls(
            param_names=param_names,
            lower=np.asarray(configs.min(axis=0), dtype=np.float64),
            upper=np.asarray(configs.max(axis=0), dtype=np.float64),
        )

    def contains(self, config: FloatArray) -> bool:
        """Whether ``config`` (P,) is inside the trust region (the ``in_domain`` flag)."""
        return bool(np.all(config >= self.lower) and np.all(config <= self.upper))

    def margin(self, config: FloatArray) -> float:
        """Signed normalized distance to the boundary — negative when outside (the OOD margin).

        Per parameter, the fractional distance to the nearer face over the box width; the region
        margin is the minimum across parameters (the tightest constraint). Positive inside,
        negative outside; a degenerate (zero-width) parameter is skipped.
        """
        width = self.upper - self.lower
        safe = np.where(width > 0.0, width, 1.0)
        to_lower = (config - self.lower) / safe
        to_upper = (self.upper - config) / safe
        per_param = np.minimum(to_lower, to_upper)
        active = width > 0.0
        return float(per_param[active].min()) if active.any() else 0.0

    def to_report_trust_region(self) -> TrustRegion:
        """Project into the SURR-01 :class:`~astro_mine.surrogate.report.TrustRegion`."""
        return TrustRegion(
            bounds={
                name: Bound(low=float(lo), high=float(hi))
                for name, lo, hi in zip(self.param_names, self.lower, self.upper, strict=True)
            }
        )
