"""RM-P0-SIM-03 — the granular engine (reduced-order excavation).

An excavator idles until commanded, then removes regolith at a capped rate up to a target
volume, accruing excavated mass and drawing battery for the work — exactly (BIT_EXACT) and
behind the waist.
"""

from __future__ import annotations

import math

from astro_mine.core.messages.enums import (
    ActionKind,
    ExcavationPattern,
    ExcavationTool,
    TaskKind,
)
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ExcavateTask,
    ModeCommand,
    TaskDirective,
    Vec3,
    Volume,
)
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import RegimeEngine
from astro_mine.sim.engines.granular import (
    GRANULAR_ENGINE_DESCRIPTOR,
    granular_engine_factory,
)
from astro_mine.sim.runtime import AgentSpec, GranularDynamics, RngStreams, Scenario


def _scenario(**dyn: object) -> Scenario:
    params: dict[str, object] = {
        "regolith_density_kg_m3": 1500.0,
        "specific_energy_j_per_m3": 1.0e5,
        "max_dig_rate_m3_s": 0.01,
    }
    params.update(dyn)
    return Scenario(
        name="dig",
        agents=(
            AgentSpec(
                agent_id="digger",
                battery_soc_j=1.0e6,
                dynamics=GranularDynamics(**params),  # type: ignore[arg-type]
            ),
            AgentSpec(agent_id="other"),  # kinematic default — skipped
        ),
    )


def _excavate(target: float | None) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id="digger",
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.EXCAVATE,
                    excavate=ExcavateTask(
                        region=Volume(
                            frame="MOON_ME",
                            center_m=Vec3(x=0.0, y=0.0, z=0.0),
                            dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
                        ),
                        tool=ExcavationTool.BUCKET,
                        pattern=ExcavationPattern.TRENCH,
                        target_volume_m3=target,
                    ),
                ),
            )
        ]
    )


def test_factory_builds_only_granular_agents() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    assert set(engine.export_coupling_state().by_agent) == {"digger"}


def test_idle_until_commanded() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    engine.advance(1.0)  # no excavate task → no dig
    assert engine.excavated_volume_m3("digger") == 0.0
    assert engine.export_coupling_state().by_agent["digger"].battery_soc_j == 1.0e6


def test_digs_at_capped_rate_with_mass_and_energy() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_excavate(0.05))
    engine.advance(1.0)  # rate 0.01 m³/s * 1 s
    assert math.isclose(engine.excavated_volume_m3("digger"), 0.01, rel_tol=1e-12)
    assert math.isclose(engine.excavated_mass_kg("digger"), 0.01 * 1500.0, rel_tol=1e-12)
    soc = engine.export_coupling_state().by_agent["digger"].battery_soc_j
    assert soc is not None and math.isclose(
        soc, 1.0e6 - 1.0e5 * 0.01, rel_tol=1e-12
    )  # work = 1000 J


def test_respects_target_volume() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_excavate(0.025))  # 2.5 ticks of capacity
    for _ in range(10):
        engine.advance(1.0)
    assert math.isclose(
        engine.excavated_volume_m3("digger"), 0.025, rel_tol=1e-12
    )  # stops at target


def test_none_target_digs_continuously() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_excavate(None))
    for _ in range(4):
        engine.advance(1.0)
    assert math.isclose(engine.excavated_volume_m3("digger"), 0.04, rel_tol=1e-12)


def test_mode_command_sets_mode() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(agent_id="digger", kind=ActionKind.MODE, mode=ModeCommand(mode="excavating"))
            ]
        )
    )
    assert engine.export_coupling_state().by_agent["digger"].mode == "excavating"


def test_excavation_is_bit_exact_deterministic() -> None:
    a = granular_engine_factory(_scenario(), RngStreams(0))
    b = granular_engine_factory(_scenario(), RngStreams(0))
    for engine in (a, b):
        engine.apply_actions(_excavate(0.037))
        for _ in range(6):
            engine.advance(1.0)
    assert a.excavated_volume_m3("digger") == b.excavated_volume_m3("digger")
    assert (
        a.export_coupling_state().by_agent["digger"].battery_soc_j
        == b.export_coupling_state().by_agent["digger"].battery_soc_j
    )


def test_descriptor_declares_surface_massmodel_bit_exact() -> None:
    d = GRANULAR_ENGINE_DESCRIPTOR
    assert d.regimes == (Regime.SURFACE,)
    assert d.frames == (MOON_BODY_FIXED,)
    assert d.fidelity.tier is FidelityTier.MASSMODEL
    assert d.determinism_class is DeterminismClass.BIT_EXACT
    assert d.to_manifest().kind is PluginKind.REGIME_ENGINE


def test_engine_satisfies_regime_engine_protocol() -> None:
    assert isinstance(granular_engine_factory(_scenario(), RngStreams(0)), RegimeEngine)


def test_coupling_state_round_trips() -> None:
    engine = granular_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(_excavate(0.02))
    engine.advance(1.0)
    snapshot = engine.export_coupling_state()
    engine.advance(1.0)
    engine.import_coupling_state(snapshot)
    restored = engine.export_coupling_state().by_agent["digger"]
    assert restored.battery_soc_j == snapshot.by_agent["digger"].battery_soc_j
    assert restored.pose == snapshot.by_agent["digger"].pose
