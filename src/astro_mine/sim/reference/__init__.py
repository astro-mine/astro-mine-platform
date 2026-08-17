# SPDX-License-Identifier: Apache-2.0
"""A small, offline reference environment — the always-works quickstart tier.

The anchor scenario is a *benchmark*: ~461 MB across nine pins from a private registry, over a
43 200-tick lunar month. Nothing shipped in this package let a consumer obtain a runnable
:class:`~astro_mine.sim.runtime.scenario.Scenario` without either writing one by hand or
downloading that. This module is the missing five-minute path (conventions.md §7 tier 1, the
local tier that MUST always work; learn.md §7): a synthetic three-agent scenario carried as
package data, plus the Core-typed handles a consumer needs to wrap it.

**No content, no store, no registry, no network.** The scenario pins nothing, and
:class:`~astro_mine.sim.runtime.episode.Simulator` falls back to
:class:`~astro_mine.sim.power_thermal.ReferenceWorldProvider` when no world is injected, so
construction needs only this package.

**The seam is Core-typed, and deliberately so.** ``make_reference_env`` returns a Core
:class:`~astro_mine.core.env.protocol.Environment`; ``reference_assets`` returns Core
:class:`~astro_mine.core.sadf.model.Asset` documents. A multi-agent RL consumer needs both — an
environment to step and the SADF to derive per-agent spaces from — and
``make_reference_env_and_assets`` hands over exactly that pair, so the consumer builds its own
wrapper without this package importing it (`conventions.md` §1.1). The bridge is the
``module:attr`` string the consumer already resolves at runtime, which is a static dependency for
neither side. Contrast :mod:`astro_mine.sim.bench`, which *must* import Bench because it satisfies
Bench-owned protocol types; nothing here names a non-Core type.

**Rewards are the consumer's.** ``StepResult.rewards`` stays empty: reward shaping is a training
concern, not a physics one, and a consumer supplies its own function over these observations.

**Actions matter — including the ones a learned policy actually emits.** The two surface agents
declare :class:`~astro_mine.sim.runtime.MobilityDynamics` and the environment is built with
:func:`~astro_mine.sim.coupling.coupled_engine_factory`, so they route to the mobility engine,
which honours a ``MODE`` command, a ``VELOCITY``
:class:`~astro_mine.core.messages.model.ActuatorCommand`, **and** a
:class:`~astro_mine.core.messages.model.GotoTask`.

That last one is load-bearing and easy to get wrong. Sim's default ``KinematicEngine`` honours
only ``MODE`` and ``VELOCITY`` — it silently ignores a ``TASK`` — while an RL adapter's mobility
modality encodes to ``ActionKind.TASK`` with a ``GotoTask`` and never emits ``VELOCITY`` at all.
On the kinematic engine such a policy therefore moves nothing, every pose-derived reward is flat
with respect to it, and training is vacuous while appearing to run. Two action vocabularies
coupled by convention with nothing validating the join — the same shape as the Fleet-mode /
extraction-set gap. The tests pin *both* action kinds for exactly that reason.
"""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

from astro_mine.core.sadf.enums import CapabilityTag, CommsBand, NodeRole
from astro_mine.core.sadf.model import Asset, Comms, Identity
from astro_mine.sim.coupling import coupled_engine_factory
from astro_mine.sim.runtime.episode import Simulator
from astro_mine.sim.runtime.scenario import Scenario

if TYPE_CHECKING:
    from astro_mine.core.env.model import AgentId

__all__ = [
    "REFERENCE_SCENARIO_FILE",
    "load_reference_scenario",
    "make_reference_env",
    "make_reference_env_and_assets",
    "reference_assets",
]

#: The packaged scenario document, resolved through :mod:`importlib.resources` so it is readable
#: from an installed wheel and not only from a source checkout.
REFERENCE_SCENARIO_FILE = "scenario.json"

_SADF_INTERFACE_VERSION = "0.1.0"


def load_reference_scenario() -> Scenario:
    """Load and validate the packaged reference :class:`Scenario`."""
    text = files(__package__).joinpath(REFERENCE_SCENARIO_FILE).read_text(encoding="utf-8")
    return Scenario.from_json(text)


def reference_assets() -> dict[AgentId, Asset]:
    """The SADF documents describing the reference scenario's agents.

    Heterogeneity here is by **declared capability**, not by sensor suite: a consumer derives an
    agent's action modalities from its capabilities, so a wheeled rover, a tracked excavator and
    an orbiting relay get genuinely different spaces from these three documents alone.

    No asset declares a sensing capability, and no agent in the scenario declares a sensor. That
    pairing is deliberate: a sensing block appears in a derived observation space only when an
    asset declares both a sensing capability *and* a matching sensor, and a sensor Sim cannot fill
    without a resource field or a world bundle would render invalid on every tick. The smoke tier
    exercises the loop, not the science — so it declares neither half rather than half of each.

    The keys match :func:`load_reference_scenario`'s agent ids exactly; the two are asserted
    consistent by test, because they are coupled by convention with nothing else to catch a drift.
    """
    interfaces = {"sadf": _SADF_INTERFACE_VERSION}
    return {
        "rover": Asset(
            identity=Identity(
                id="reference-rover",
                name="Reference Prospecting Rover",
                version="0.1.0",
                kind="rover",
            ),
            capabilities=[CapabilityTag.MOBILITY_WHEELED],
            core_interface_versions=dict(interfaces),
            root_frame="body",
        ),
        "excavator": Asset(
            identity=Identity(
                id="reference-excavator",
                name="Reference Excavator",
                version="0.1.0",
                kind="excavator",
            ),
            capabilities=[CapabilityTag.MOBILITY_TRACKED, CapabilityTag.EXCAVATION_BUCKET],
            core_interface_versions=dict(interfaces),
            root_frame="body",
        ),
        "relay": Asset(
            identity=Identity(
                id="reference-relay",
                name="Reference Relay Orbiter",
                version="0.1.0",
                kind="orbiter",
            ),
            # Comms only, and deliberately no mobility tag: a mobility capability would give the
            # relay a `goto` action modality it has no mobility dynamics to execute — advertising
            # a control a consumer cannot actually exercise, which is the defect this module's
            # tests exist to prevent, in miniature.
            capabilities=[CapabilityTag.COMMS_RELAY],
            core_interface_versions=dict(interfaces),
            root_frame="body",
            comms=[
                Comms(
                    name="relay-radio",
                    band=CommsBand.S_BAND,
                    node_role=NodeRole.SPACE,
                    relay=True,
                )
            ],
        ),
    }


def make_reference_env() -> Simulator:
    """Build the reference environment — a Core :class:`Environment`, ready to ``reset()``.

    Offline and dependency-light: no content store, no registry, no network, no extra.

    Built with :func:`coupled_engine_factory` so each agent runs the dynamics it declares. That is
    not a detail: the default factory routes *every* agent to the kinematic engine regardless of
    its ``dynamics`` block, and the kinematic engine ignores ``TASK`` actions — so a policy whose
    mobility modality encodes to a ``GotoTask`` would move nothing.
    """
    return Simulator(load_reference_scenario(), engine_factory=coupled_engine_factory())


def make_reference_env_and_assets() -> tuple[Simulator, dict[AgentId, Asset]]:
    """The reference environment paired with the SADF describing its agents.

    This is the single symbol a multi-agent RL consumer resolves through its ``module:attr``
    env-factory seam: it needs the Core :class:`Environment` to step *and* the Core
    :class:`~astro_mine.core.sadf.model.Asset` map to derive per-agent observation/action spaces
    from, and returning both keeps the contract expressible in Core types alone.
    """
    return make_reference_env(), reference_assets()
