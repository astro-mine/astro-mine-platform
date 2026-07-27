"""Engine-adapter framework — the ``RegimeEngine`` plugin contract (RM-P0-SIM-02).

The plugin seam that routes physics engines behind the Core Environment waist. A
:class:`RegimeEngine` is the adapter every backend (orbital, mobility/contact,
manipulation, granular — RM-P0-SIM-03) implements: it owns its dynamical state and the
stepping core drives it through a small, stable contract — the **coupling triad**
``advance`` / ``export_coupling_state`` / ``import_coupling_state``, the per-step
``apply_actions`` actuation hook, and a ``retire`` lifecycle hook — plus an introspectable
:class:`EngineDescriptor` declaring the engine's
**frames**, **determinism class**, and **fidelity descriptor** (sim.md §3, §11).

The descriptor maps onto the Core plugin manifest (RM-P0-CORE-05): ``kind`` is
:attr:`~astro_mine.core.registry.PluginKind.REGIME_ENGINE`, the determinism class and the
served regimes are first-class manifest fields, and the **fidelity descriptor** (which
Core deliberately does not schematize) rides in the manifest's open ``attributes`` map
exactly as Core prescribes — so an engine is discovered, version-negotiated, and
capability-gated through :class:`~astro_mine.sim.engines.registry.EngineRegistry` without
widening the waist.

Coupling state crosses a boundary as a :class:`CouplingState` of per-agent Core
:class:`~astro_mine.core.messages.model.StateSample`\\ s — frame-explicit by construction
(every sample names its :class:`~astro_mine.core.units.ReferenceFrame`). The multi-domain
coupler (RM-P0-SIM-04) is the surface's primary consumer; here it is the engine's I/O
type, and the stepping core also reads it to render observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from astro_mine.core.registry import PluginKind, PluginManifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from astro_mine.core.messages.model import ActionBatch, StateSample
    from astro_mine.core.registry import Signature
    from astro_mine.core.sadf.enums import (
        CapabilityTag,
        DeterminismClass,
        FidelityTier,
        Regime,
        SurrogatePhysicsDomain,
    )
    from astro_mine.core.units import ReferenceFrame

__all__ = [
    "ENGINE_CORE_INTERFACES",
    "CouplingState",
    "EngineDescriptor",
    "FidelityDescriptor",
    "RegimeEngine",
]

#: The Core interfaces a regime engine is built against — the input to load-time version
#: negotiation (RM-P0-CORE-07). An engine produces Core ``messages``
#: (:class:`~astro_mine.core.messages.model.StateSample`); that is the negotiated
#: contract. ``units`` frames are shared *primitives*, not a version-negotiated interface
#: (see :data:`astro_mine.core.compat.CORE_INTERFACE_VERSIONS`), so they are absent.
ENGINE_CORE_INTERFACES: dict[str, str] = {"messages": "0.1.0"}


@dataclass(frozen=True, slots=True)
class FidelityDescriptor:
    """The fidelity tier an engine runs at, plus (for a surrogate tier) the physics
    domain it substitutes.

    This is the **fidelity descriptor** the multi-fidelity scheduler (RM-P0-SIM-05) will
    consume; here it is declared, not yet selected on. Core does not schematize it, so it
    rides in the manifest's open ``attributes`` map (:meth:`as_attributes`)."""

    tier: FidelityTier
    surrogate_domain: SurrogatePhysicsDomain | None = None

    def as_attributes(self) -> dict[str, Any]:
        """The JSON-able form carried at ``PluginManifest.attributes['fidelity']``."""
        data: dict[str, Any] = {"tier": self.tier.value}
        if self.surrogate_domain is not None:
            data["surrogate_domain"] = self.surrogate_domain.value
        return data


@dataclass(frozen=True, slots=True)
class EngineDescriptor:
    """An engine's introspectable self-declaration: identity, the regimes it serves, the
    reference frames it operates in, its determinism class, and its fidelity descriptor.

    :meth:`to_manifest` renders it as a Core :class:`~astro_mine.core.registry.PluginManifest`
    (kind ``regime_engine``) so the engine is registered, version-negotiated, and
    capability-gated through Core's registry (RM-P0-CORE-05). The ``frames`` and
    ``fidelity`` — which Core does not schematize — ride in the manifest's open
    ``attributes`` map; ``determinism_class`` and ``regimes`` are first-class manifest
    fields."""

    name: str
    version: str
    regimes: tuple[Regime, ...]
    frames: tuple[ReferenceFrame, ...]
    determinism_class: DeterminismClass
    fidelity: FidelityDescriptor
    capability_tags: tuple[CapabilityTag, ...] = ()
    core_interfaces: Mapping[str, str] = field(default_factory=lambda: dict(ENGINE_CORE_INTERFACES))

    def to_manifest(self, *, signature: Signature | None = None) -> PluginManifest:
        """Render this descriptor as a Core plugin manifest for registry gating."""
        return PluginManifest(
            name=self.name,
            version=self.version,
            kind=PluginKind.REGIME_ENGINE,
            core_interfaces=dict(self.core_interfaces),
            capability_tags=list(self.capability_tags),
            determinism_class=self.determinism_class,
            regimes=list(self.regimes),
            attributes={
                "fidelity": self.fidelity.as_attributes(),
                "frames": [f.model_dump(mode="json") for f in self.frames],
            },
            signature=signature,
        )


@dataclass(frozen=True, slots=True)
class CouplingState:
    """Per-agent dynamical state crossing a coupling boundary at one instant.

    A frame-explicit snapshot — every :class:`~astro_mine.core.messages.model.StateSample`
    names its :class:`~astro_mine.core.units.ReferenceFrame` — exchanged by the coupling
    triad (:meth:`RegimeEngine.export_coupling_state` /
    :meth:`RegimeEngine.import_coupling_state`). ``sim_time_s`` is the engine-local elapsed
    time the snapshot is valid at. The multi-domain coupler (RM-P0-SIM-04) is the primary
    consumer; the stepping core also reads it to render observations."""

    sim_time_s: float
    samples: tuple[StateSample, ...]
    #: Cumulative excavated regolith mass (kg) per agent, for engines that dig (#64).
    #:
    #: A **Sim-owned** channel, deliberately not a Core ``StateSample`` field: excavated mass is
    #: an internal quantity of the value chain, not something an agent observes about itself, and
    #: ``StateSample`` is in the Cap'n Proto hot-path family where a new field moves
    #: ``SCHEMA_DIGEST`` and re-pins every scenario's ``core_schema_digest``. It rides here instead
    #: — the coupling boundary the engines already cross every tick — so
    #: :class:`~astro_mine.sim.logistics.Material` can be accrued where it was dug.
    #:
    #: Cumulative rather than per-tick so it is idempotent under replay: a consumer differences it.
    #: Engines that do not dig contribute nothing, and the mapping is empty for all of them.
    excavated_kg: Mapping[str, float] = field(default_factory=dict)

    @property
    def by_agent(self) -> dict[str, StateSample]:
        """The samples keyed by ``agent_id`` (insertion order preserved)."""
        return {s.agent_id: s for s in self.samples}


@runtime_checkable
class RegimeEngine(Protocol):
    """The adapter every physics engine implements to route behind the waist.

    The stepping core owns the loop; the engine owns its dynamical state and is driven
    through the **coupling triad** (``advance`` / ``export_coupling_state`` /
    ``import_coupling_state``) plus a ``retire`` lifecycle hook. No engine *type* ever
    leaks past this contract — consumers see only Core messages (sim.md §2.1: "no engine
    ever leaks through the waist"). Determinism is the engine's declared class
    (:attr:`EngineDescriptor.determinism_class`), enforced in CI per engine (sim.md §11)."""

    @property
    def descriptor(self) -> EngineDescriptor:
        """This engine's introspectable frames / determinism class / fidelity."""
        ...

    def apply_actions(self, actions: ActionBatch) -> None:
        """Actuate this step's commands before :meth:`advance`.

        The stepping core hands the whole :class:`~astro_mine.core.messages.model.ActionBatch`
        to every live engine; an engine applies only the actions addressed to agents it owns
        and ignores the rest, so an empty batch is a no-op. Actuation sets the agents'
        command state (velocity/goto, joint setpoints, dig directives, …); the subsequent
        :meth:`advance` integrates under it
        (:mod:`astro_mine.sim.engines.actuation` is the shared dispatch helper)."""
        ...

    def advance(self, dt_s: float) -> None:
        """Integrate every live agent forward by ``dt_s`` seconds (SI)."""
        ...

    def export_coupling_state(self) -> CouplingState:
        """Snapshot the live agents' state at the coupling boundary."""
        ...

    def import_coupling_state(self, state: CouplingState) -> None:
        """Overwrite live agents' state from a boundary snapshot — the inverse of
        :meth:`export_coupling_state` for cross-engine handoff. Samples for agents this
        engine does not own are ignored."""
        ...

    def retire(self, agent_ids: Iterable[str]) -> None:
        """Drop terminated agents so they are neither advanced nor exported again."""
        ...
