"""SweepSpec -- a parameter space over a base JobSpec.

A :class:`SweepSpec` describes many related jobs as one object: a base
:class:`~astro_mine.cloud.submission.jobspec.JobSpec` plus a parameter space
(``cloud.md`` §3). :meth:`SweepSpec.expand` enumerates it **deterministically** into concrete
JobSpecs -- a Cartesian ``grid``, a seeded ``random`` sample, or a low-discrepancy
``halton`` quasi-random sample (our pure stand-in for Sobol; adaptive Optuna / Ray-Tune
search is an ``engines/`` hook, not in-process expansion). Each variant injects its parameter
assignment into the job's ``env`` and derives a distinct, reproducible seed, so a sweep run
reproduces exactly (``cloud.md`` §2 principle 4). The sweep compiles to an Argo fan-out in
:mod:`astro_mine.cloud.engines.argo`.

Backlog: RM-P1-CLOUD-02 -- astro-mine-cloud#13
"""

from __future__ import annotations

import itertools
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.cloud.submission.jobspec import JobSpec

__all__ = ["ParamValue", "SweepSpec"]

#: A single grid parameter value. Scalars only -- they inject into ``env`` as strings.
ParamValue = int | float | str | bool

#: Env var recording a variant's 0-based index within the sweep.
SWEEP_INDEX_ENV = "ASTRO_MINE_SWEEP_INDEX"


def _radical_inverse(index: int, base: int) -> float:
    """Van der Corput radical inverse of *index* in *base* -- the Halton building block."""
    result, fraction = 0.0, 1.0 / base
    while index > 0:
        index, digit = divmod(index, base)
        result += digit * fraction
        fraction /= base
    return result


# First primes seed the Halton bases, one per swept dimension.
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


class SweepSpec(BaseModel):
    """A base JobSpec plus a parameter space that expands to concrete JobSpecs."""

    model_config = ConfigDict(extra="forbid")

    base: JobSpec
    method: Literal["grid", "random", "halton"] = "grid"
    #: ``grid``: parameter name -> the discrete values to sweep (Cartesian product).
    grid: dict[str, list[ParamValue]] = Field(default_factory=dict)
    #: ``random``/``halton``: parameter name -> ``(low, high)`` continuous range.
    ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    #: ``random``/``halton``: number of samples to draw.
    samples: int | None = None
    #: Sampling seed (``random``/``halton``) -- makes the drawn points reproducible.
    seed: int = 0
    #: Optional Argo ``parallelism`` cap so a huge fan-out does not stampede the scheduler.
    max_parallel: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_space(self) -> SweepSpec:
        if self.method == "grid":
            if not self.grid:
                raise ValueError("method 'grid' requires a non-empty `grid`")
            if self.ranges or self.samples is not None:
                raise ValueError("method 'grid' uses `grid`, not `ranges`/`samples`")
            for name, values in self.grid.items():
                if not values:
                    raise ValueError(f"grid parameter {name!r} has no values")
        else:  # random / halton
            if not self.ranges:
                raise ValueError(f"method {self.method!r} requires `ranges`")
            if self.samples is None or self.samples <= 0:
                raise ValueError(f"method {self.method!r} requires a positive `samples`")
            if self.grid:
                raise ValueError(f"method {self.method!r} uses `ranges`, not `grid`")
            if len(self.ranges) > len(_PRIMES):
                raise ValueError(f"halton supports up to {len(_PRIMES)} dimensions")
            for name, (low, high) in self.ranges.items():
                if not low < high:
                    raise ValueError(f"range for {name!r} must have low < high, got {(low, high)}")
        return self

    def size(self) -> int:
        """Number of concrete jobs this sweep expands to."""
        if self.method == "grid":
            total = 1
            for values in self.grid.values():
                total *= len(values)
            return total
        assert self.samples is not None  # guaranteed by the validator
        return self.samples

    def _assignments(self) -> list[dict[str, ParamValue]]:
        """Enumerate the parameter assignments, one per variant, deterministically."""
        if self.method == "grid":
            names = sorted(self.grid)
            return [
                dict(zip(names, combo, strict=True))
                for combo in itertools.product(*(self.grid[name] for name in names))
            ]
        names = sorted(self.ranges)
        assert self.samples is not None
        if self.method == "random":
            rng = random.Random(self.seed)
            return [
                {name: rng.uniform(*self.ranges[name]) for name in names}
                for _ in range(self.samples)
            ]
        # halton: skip index 0 (which maps every dimension to 0.0) for a better spread.
        out: list[dict[str, ParamValue]] = []
        for i in range(1, self.samples + 1):
            point: dict[str, ParamValue] = {}
            for dim, name in enumerate(names):
                low, high = self.ranges[name]
                point[name] = low + (high - low) * _radical_inverse(i + self.seed, _PRIMES[dim])
            out.append(point)
        return out

    def _variant(self, index: int, assignment: dict[str, ParamValue]) -> JobSpec:
        env = {
            **self.base.env,
            **{name: str(value) for name, value in assignment.items()},
            SWEEP_INDEX_ENV: str(index),
        }
        # Derive a distinct, reproducible per-variant seed from the base seed and index.
        seed = None if self.base.seed is None else self.base.seed + index
        return self.base.model_copy(update={"env": env, "seed": seed})

    def expand(self) -> list[JobSpec]:
        """Enumerate the sweep into concrete, deterministically-seeded JobSpecs."""
        return [self._variant(i, a) for i, a in enumerate(self._assignments())]
