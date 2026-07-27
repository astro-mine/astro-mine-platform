"""A learned (ONNX) policy, shielded end-to-end on the anchor (RM-P1-GUARD-05; issue #5).

The Phase-1 Guard exit criterion — *"a Learn policy runs **shielded** in Sim on the anchor
scenario; zero hard-constraint violations under adversarial test"* (roadmap
`phase-1-autonomy-studio.md`) — is demonstrated at two levels:

- **The always-on gate** (no optional deps): a Core :class:`~astro_mine.core.policy.OnnxPolicy` — an
  exported ``PolicyPackage`` bound to a deterministic host ``infer`` — wrapped by
  :class:`~astro_mine.guard.wrap.PolicyShield` and driven through the in-repo double-integrator
  rollout on the anchor. The learned policy is deliberately unsafe (it thrusts straight at the
  lander-zone keep-out); shielded, the rollout yields **zero violations**.

- **The real-Sim rollout** (:func:`test_real_onnx_rollout_against_real_sim`, ``sim``-marked): a
  **genuine ONNX graph**, executed by **ONNX Runtime**, wrapped by the shield, and stepped through
  a **real** :mod:`astro_mine.sim` episode — Sim's own :class:`~astro_mine.sim.runtime.Simulator` /
  :func:`~astro_mine.sim.runtime.run_episode`, its own kinematic engine, its own
  :class:`~astro_mine.sim.runtime.Trace`. Nothing is stubbed: the plant is Sim's, the policy is a
  real graph, and the safety is the Rust TCB's. The violations are then scored **from Sim's
  trace**, not from Guard's own bookkeeping, so the claim is checked against an independent record
  of what actually happened.

  This is the test that used to be a placeholder — it ``importorskip``ed ``onnxruntime`` and then
  unconditionally ``pytest.skip``ed. It is now real. It stays ``sim``-marked (deselected from the
  default CI job, per ``pyproject.toml``) because it needs the ``[sim]`` extra; the dedicated
  ``sim-e2e`` CI job installs that extra and runs it, so it is a **gate**, not a hope.

  Sim's reference engine actuates **VELOCITY** setpoints — which is exactly why this rollout can
  exist at all: shielding beyond ``EFFORT`` (RM-P1-GUARD-03's action gate) is what lets Guard sit
  between a real policy and a real plant that speaks a real actuation channel.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand
from astro_mine.core.policy import DecisionContext, OnnxPolicy, check_policy
from astro_mine.core.policy.model import ModelRef, PolicyPackage
from astro_mine.guard.audit.sink import CollectingSink
from astro_mine.guard.falsify import (
    DEFAULT_DT,
    DEFAULT_U_MAX,
    WorstCaseAdversary,
    anchor_initial_state,
    shielded_rollout,
    shielded_violations,
)
from astro_mine.guard.models import compile_anchor
from astro_mine.guard.wrap import CoreConfig, MappingSignalResolver, PolicyShield
from tests.guard.conftest import SAFE_SIGNALS, make_observation

COARSE_PERIOD_S = 120_960.0
_U_MAX = 20.0


def _aggressive_infer(observations, context):  # type: ignore[no-untyped-def]
    """A deterministic stub 'learned' policy: full thrust straight at the lander-zone sphere centre
    (the origin) — exactly the action the shield must veto."""
    actions = []
    for agent_id, obs in observations.items():
        t = obs.self_state.pose.translation_m
        vec = [-t.x, -t.y, -t.z]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        setpoint = [_U_MAX * v / norm for v in vec]
        actions.append(
            Action(
                agent_id=agent_id,
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="body", control_mode=ControlMode.EFFORT, setpoint=setpoint
                ),
            )
        )
    return ActionBatch(actions=actions)


def _onnx_policy() -> OnnxPolicy:
    package = PolicyPackage(
        name="anchor-prospector-controller",
        version="0.1",
        onnx_model=ModelRef(digest=content_hash_json({"stub": "policy"})),
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
    )
    return OnnxPolicy(package, _aggressive_infer)


def test_onnx_policy_shielded_yields_zero_violations() -> None:
    compiled = compile_anchor(sample_period_s=COARSE_PERIOD_S)
    sink = CollectingSink()
    shield = PolicyShield(_onnx_policy(), compiled, sink=sink, core_config=CoreConfig())
    # A zero-drain worst-case adversary supplies only the (safe, constant) signals and no external
    # disturbance — the *action* under test is the learned OnnxPolicy's, corrected by the shield.
    adversary = WorstCaseAdversary(compiled, drain_fraction=0.0)
    steps = shielded_rollout(
        shield, adversary, initial=anchor_initial_state(), horizon=150, sink=sink
    )
    assert shielded_violations(steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT) == []
    # the learned policy really did drive toward the keep-out: the shield had to intervene
    interventions = [s for s in steps if s.verdict is not None and s.verdict.intervention != "none"]
    assert interventions, "the aggressive policy was never corrected — the test is vacuous"


def test_shielded_onnx_policy_passes_the_core_policy_contract() -> None:
    compiled = compile_anchor(sample_period_s=COARSE_PERIOD_S)
    shield = PolicyShield(_onnx_policy(), compiled, core_config=CoreConfig())
    obs = make_observation("rover", position=(45.0, 0.0, 25.0))
    # The shielded learned policy is itself a well-formed, Sim-consumable Core Policy.
    batch = check_policy(shield, {"rover": obs}, DecisionContext())
    assert batch.actions
    assert all(math.isfinite(x) for x in batch.actions[0].actuator.setpoint)


# --- the real-Sim, real-ONNX rollout (RM-P1-GUARD-03/-05; the Phase-1 Guard exit criterion) ----

#: The lander-zone keep-out the anchor SafetySpec declares (sphere r = 30 m, margin 3 m at the
#: origin of the body-fixed frame). Re-derived here so the rollout is scored against the *authored
#: contract*, independently of anything Guard computed at run time.
_KEEPOUT_RADIUS_M = 30.0 + 3.0

#: The reviewed traverse-speed ceiling (`c_traverse_speed`), which the shield now enforces on the
#: *commanded* velocity as well as on the measured signal.
_V_MAX_MPS = 0.1


def _homing_onnx_bytes() -> bytes:
    """A real ONNX graph: ``y = W · x`` with ``W = -k·I`` — a proportional homing controller.

    Fed the agent's position, it emits a velocity straight at the origin — i.e. straight into the
    lander-zone keep-out. That is deliberate: an adversarial *learned* policy is the input the
    shield
    exists to certify (guard.md §9.1, "a learned policy is an untrusted, adversarial input")."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    gain = numpy_helper.from_array(np.diag([-0.5, -0.5, -0.5]).astype(np.float32), name="W")
    node = helper.make_node("MatMul", ["x", "W"], ["y"])
    graph = helper.make_graph(
        [node],
        "homing_controller",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[gain],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString()  # type: ignore[no-any-return]


@pytest.mark.sim
def test_real_onnx_rollout_against_real_sim() -> None:
    """A **real** ONNX policy runs **shielded** through a **real** Sim episode, with zero
    violations.

    End to end, with nothing stubbed:

    - the policy is an ONNX graph executed by ONNX Runtime, hosted behind Core's ``OnnxPolicy``;
    - the shield is Guard's ``PolicyShield`` over the Rust TCB, enforcing the anchor ``SafetySpec``;
    - the plant is ``astro_mine.sim``'s own :class:`Simulator` (its kinematic engine actuates the
      certified ``VELOCITY`` setpoints);
    - the verdict is scored from **Sim's** :class:`Trace` — the poses the simulator actually
      integrated — not from Guard's own record of what it thinks it did.

    The unshielded control is asserted too: the same policy, unwrapped, drives straight into the
    keep-out. Without that, "zero violations" would be unfalsifiable."""
    pytest.importorskip("onnx", reason="the [sim] extra is not installed")
    pytest.importorskip("onnxruntime", reason="the [sim] extra is not installed")
    pytest.importorskip("astro_mine.sim", reason="the [sim] extra is not installed")

    import numpy as np
    import onnxruntime as ort

    from astro_mine.core.env.model import AgentId
    from astro_mine.core.messages.model import Observation
    from astro_mine.sim.runtime import AgentSpec, Scenario, Trace, run_episode

    session = ort.InferenceSession(_homing_onnx_bytes(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def infer(observations: dict[AgentId, Observation], context: DecisionContext) -> ActionBatch:
        """Host-side inference: position in, velocity setpoint out. Core owns the adapter shape but
        never runs the graph — the host does (core.md §; the same contract Mind's ONNX tier
        uses)."""
        actions = []
        for agent_id in sorted(observations):
            t = observations[agent_id].self_state.pose.translation_m
            x = np.array([[t.x, t.y, t.z]], dtype=np.float32)
            (y,) = session.run(None, {input_name: x})
            actions.append(
                Action(
                    agent_id=agent_id,
                    kind=ActionKind.ACTUATOR,
                    actuator=ActuatorCommand(
                        target="base",
                        control_mode=ControlMode.VELOCITY,
                        setpoint=[float(v) for v in y[0]],
                        unit="m/s",
                    ),
                )
            )
        return ActionBatch(actions=actions)

    package = PolicyPackage(
        name="anchor-homing-controller",
        version="0.1",
        onnx_model=ModelRef(digest=content_hash_json({"graph": "homing"})),
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
    )
    learned = OnnxPolicy(package, infer)

    # The scenario: one rover starting just outside the lander-zone keep-out, in Sim's own runtime.
    start = (40.0, 0.0, 10.0)
    horizon = 60
    scenario = Scenario(
        name="guard-shielded-onnx-anchor",
        agents=(AgentSpec(agent_id="rover", initial_position_m=start, battery_soc_j=1e6),),
        dt_s=1.0,
        horizon_steps=horizon,
        seed=7,
    )

    def _poses(trace: Trace) -> list[tuple[float, float, float]]:
        """Every pose Sim actually integrated, read back off its canonical Trace."""
        poses: list[tuple[float, float, float]] = []
        for frame in trace.frames:
            for obs in frame["observations"].values():
                t = obs["self_state"]["pose"]["translation_m"]
                poses.append((t["x"], t["y"], t["z"]))
        return poses

    def _range(pose: tuple[float, float, float]) -> float:
        return math.sqrt(sum(c * c for c in pose))

    # (1) The UNSHIELDED control: the learned policy really is unsafe. Without this the
    # zero-violation
    #     claim below would be vacuous.
    control = run_episode(scenario, policy=learned, seed=7)
    breaches = [p for p in _poses(control) if _range(p) < _KEEPOUT_RADIUS_M]
    assert breaches, "the learned policy never breached the keep-out — the test is vacuous"

    # (2) The SHIELDED run: the same policy, same scenario, same seed — wrapped by the real TCB.
    compiled = compile_anchor(sample_period_s=scenario.dt_s)

    def _shield(sink: CollectingSink) -> PolicyShield:
        # A fresh shield per run: the temporal monitors are stateful across ticks, so a shield is
        # instantiated per episode exactly as every plugin is.
        return PolicyShield(
            learned,
            compiled,
            sink=sink,
            signal_resolver=MappingSignalResolver(SAFE_SIGNALS),
            core_config=CoreConfig(max_history_cap=2_000_000),
        )

    sink = CollectingSink()
    shielded = run_episode(scenario, policy=_shield(sink), seed=7)

    # ZERO hard-constraint violations, scored from Sim's own trace.
    poses = _poses(shielded)
    violations = [p for p in poses if _range(p) < _KEEPOUT_RADIUS_M]
    assert not violations, f"the shielded run entered the keep-out: {violations[:3]}"

    # The shield genuinely had to work for it (it is not passing an already-safe policy through) …
    interventions = [v for v in sink.verdicts if v.intervention != "none"]
    assert interventions, "the shield never intervened — the rollout is not exercising it"

    # … every certified command respected the reviewed 0.1 m/s traverse ceiling …
    for verdict in sink.verdicts:
        speed = math.sqrt(sum(x * x for x in verdict.certified_action))
        assert speed <= _V_MAX_MPS + 1e-6, f"certified speed {speed} exceeds the reviewed ceiling"

    # … and the run stayed reproducible: a fresh shielded episode at the same seed reproduces Sim's
    # content-addressed Trace byte-for-byte (the determinism gate — conventions.md §11, guard.md
    # §9.3).
    replay = run_episode(scenario, policy=_shield(CollectingSink()), seed=7)
    assert replay.content_hash == shielded.content_hash
