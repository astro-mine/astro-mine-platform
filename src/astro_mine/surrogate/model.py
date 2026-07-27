"""``SurrogateModel`` — the runtime seam a learned fidelity tier satisfies.

The prediction interface Sim's scheduler calls per query: a prediction, a *calibrated
per-channel uncertainty*, and a *trust-region flag*, on **every** call (surrogate.md §3,
principle 1). This is the "live" half of the error contract; the static
:class:`~astro_mine.surrogate.report.ErrorReport` the model also carries is the
"admission" half (surrogate.md §11: "Both — static ErrorReport for admission, plus live
per-query uncertainty for in-loop fallback").

A note on the Core boundary (established while implementing this issue): Core has **no**
single-transition ``predict(state, action)`` physics-step Protocol — its ``env``
interface is the multi-agent :class:`astro_mine.core.env.Environment` (``reset``/``step``
over a batch). So ``SurrogateModel`` is a Surrogate-owned contract, mirroring Core's
``Protocol`` idiom (``Environment``/``Policy``); the adapter that wraps a model so the
*assembled tier* satisfies Core's ``Environment`` for Sim is RM-P1-SURR-04, not this
issue. ``predict`` therefore operates on a domain-generic per-channel input vector
(``action`` is optional — a field-query surrogate has none), keeping the exact shape open
per the surrogate.md §11 co-design question rather than hardwiring an excavation state.

``Prediction`` is a frozen slotted dataclass, not a Pydantic model: it is a per-query
runtime return on the hot path, not a persisted wire document — the same choice Core
makes for ``StepResult`` (env/model.py). Only the persisted artifacts (``ErrorReport``,
the manifest) are Pydantic + wire types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.surrogate.report import ErrorReport

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    #: A per-particle field array, ``(N, D)`` in SI units (e.g. positions/velocities). Only a
    #: type here — the contract layer imports no numpy at runtime; a particle surrogate
    #: (RM-P1-SURR-02) fills these with real arrays, a scalar surrogate leaves them empty.
    FieldArray = npt.NDArray[np.float64]

__all__ = ["ChannelVector", "ParticleFields", "Prediction", "SurrogateModel", "SurrogateState"]

#: A per-channel numeric input/output vector in declared units — the domain-generic
#: shape ``predict`` speaks. Keys are channel names; values are SI-unit floats. Richer
#: structured Core states (``StateSample``/``Observation``) are projected to/from this by
#: the RM-P1-SURR-04 tier adapter, not by the contract.
ChannelVector = Mapping[str, float]

#: A per-particle field state: named ``(N, D)`` arrays (positions, velocities, …) — the input
#: a learned-DEM particle surrogate (RM-P1-SURR-02) steps. Added additively alongside the
#: scalar ``ChannelVector``; the scalar path is unchanged.
ParticleFields = Mapping[str, "FieldArray"]

#: What ``predict`` accepts: a scalar channel vector *or* a particle field state.
SurrogateState = ChannelVector | ParticleFields


@dataclass(frozen=True, slots=True)
class Prediction:
    """One surrogate prediction: value(s), calibrated uncertainty, and the trust flag.

    ``uncertainty`` is per output channel and in the channel's unit — never a single
    opaque scalar — so Sim's scheduler can test it against a per-channel budget (sim.md
    §3). ``in_domain`` is the synchronous trust-region membership flag; ``ood_margin`` is
    the optional signed distance to the trust-region boundary (negative = outside) that a
    drift monitor accumulates. An out-of-domain query sets ``in_domain=False`` and raised
    uncertainty rather than a confident extrapolation (surrogate.md principle 3).

    A **particle** surrogate (RM-P1-SURR-02) additionally returns per-particle ``fields``
    (e.g. ``{"position": (N,2), "velocity": (N,2)}``) with matching per-particle
    ``field_uncertainty``; a scalar surrogate leaves both empty. This is an additive
    extension — the scalar ``channels``/``uncertainty`` path is unchanged.
    """

    channels: Mapping[str, float]
    uncertainty: Mapping[str, float]
    in_domain: bool
    categoricals: Mapping[str, str] = field(default_factory=dict)
    ood_margin: float | None = None
    fields: Mapping[str, FieldArray] = field(default_factory=dict)
    field_uncertainty: Mapping[str, FieldArray] = field(default_factory=dict)


@runtime_checkable
class SurrogateModel(Protocol):
    """A learned fidelity tier: a calibrated prediction on every call, plus its bound.

    Implementations expose the static admission artifact (:attr:`error_report`) and the
    live prediction (:meth:`predict`). The two halves are the error contract Sim's
    multi-fidelity scheduler consumes to admit the tier within budget and fall back to
    high fidelity on an out-of-budget or out-of-domain query. Being ``runtime_checkable``,
    a duck-typed model passes :func:`isinstance` structurally — the seam is behavioural,
    not a base class to inherit.
    """

    @property
    def error_report(self) -> ErrorReport:
        """The static, calibrated :class:`ErrorReport` this model ships with (its bound)."""
        ...

    def predict(self, state: SurrogateState, action: SurrogateState | None = None) -> Prediction:
        """Predict the next state / field value(s) with calibrated per-channel uncertainty.

        ``state`` is a scalar :data:`ChannelVector` or a :data:`ParticleFields` state (a
        learned-DEM surrogate steps the latter). Returns a :class:`Prediction` on every
        call; ``action`` is omitted for a field-query (non-dynamical) surrogate. Never
        returns a confident extrapolation for an out-of-trust-region query — it lowers
        ``in_domain`` and raises uncertainty.
        """
        ...
