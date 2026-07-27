# Architecture

`astro-mine-learn` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/learn.md`](https://github.com/astro-mine/docs/blob/main/architecture/learn.md):
  purpose, principles, runtime, data, integration, security, and roadmap alignment for
  this package (the multi-agent RL toolkit — Gymnasium/PettingZoo env wrappers over the
  Core Environment API, CTDE baselines, curricula, scale-out training, and ONNX policy
  export).
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

## Status

The Phase-1 `RM-P1-LEARN-01…06` surface is **implemented**, against the Core `v0.2.0` tag:

| Deliverable | Where |
|---|---|
| `RM-P1-LEARN-01` — `SwarmEnv` adapter (Gymnasium/PettingZoo over the Core Environment API) | `envs/adapter/` |
| `RM-P1-LEARN-02` — the declarative `CommsModel` (gate → budget → drop → delay) | `envs/comms/` |
| `RM-P1-LEARN-03` — baselines: IPPO, MAPPO, QMIX + the `comms_ppo` comms-learning track | `algos/`, `models/` |
| `RM-P1-LEARN-04` — tier-1 training, KubeRay scale-out, GPU-vectorized (JAX) rollout | `train/`, `envs/vector/` |
| `RM-P1-LEARN-05` — `PolicyPackage` ONNX export (feed-forward, **recurrent**, comms-learning) + the ONNX-Runtime equivalence gate | `export/` |
| `RM-P1-LEARN-06` — honest-eval harness: held-out splits, seed sweeps, comms-stress curves | `eval/` |
| learn.md §3 `curriculum/` — staged difficulty, domain randomization, automatic-curriculum plugin seam | `curriculum/` |
| learn.md §3 `track/` — MLflow-default experiment tracking + provenance capture | `track/` |

Every heavy runtime (Ray RLlib + Torch, ONNX/ONNX-Runtime, pyarrow, JAX, MLflow) lives behind an
optional extra, so the base wheel stays a lightweight env-adapter library and the narrow-waist
surface (registry, contracts, produced-policy manifests) imports with none of them.

**Deferred → Phase 2** (per the roadmap): automatic curricula (the *interface* ships here; the
PLR/teacher-student realizations do not), learned allocation heuristics for Allocate, and
sim-to-real-aware training validated on terrestrial analogs.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).
