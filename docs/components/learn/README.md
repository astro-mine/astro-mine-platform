# astro-mine-learn

**Multi-agent reinforcement-learning toolkit for [Astro-Mine](https://github.com/astro-mine).**
Gymnasium/PettingZoo adapters over the Core Environment API (partial observability and
comms-limited links, first class), a CTDE baseline suite (MAPPO/IPPO/QMIX) plus a
comms-learning research track, staged curricula and domain randomization, MLflow-backed
experiment tracking, Ray-based scale-out and GPU-vectorized rollout, and portable ONNX policy
export. Learn is a *consumer* of Core contracts — it never widens the waist.

> **Status:** Phase 1 — the `RM-P1-LEARN-01…06` surface is **implemented**: the `SwarmEnv`
> adapter and `CommsModel`, the four baselines, tier-1/KubeRay/GPU-vectorized executors,
> `PolicyPackage` ONNX export with the equivalence gate, the honest-eval harness, curricula, and
> experiment tracking. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/learn.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

> **Where the commands live.** This package ships no console scripts. Learn's commands are
> `astro-mine learn <flags>` (a flat group — no subcommand), provided by
> [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli) — a separate distribution that
> depends on this one. There is one executable and one grammar
> ([`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
> [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md)); the earlier
> `astro-mine-learn` binary and its bare alias are both retired.

## Layout

```
src/astro_mine/learn/
├── envs/          # Core Environment API → Gymnasium/PettingZoo; CommsModel; vectorized/batched
├── algos/         # MARL plugins: IPPO, MAPPO, QMIX, comms_ppo (+ the Learn-internal registry)
├── models/        # policy/value nets: MLP, GRU core, centralized critics, message modules
├── curriculum/    # staged difficulty, domain randomization, automatic-curriculum plugin seam
├── train/         # rollout↔learner loop; Local / KubeRay / Vector executors; the CLI
├── eval/          # held-out envs, seed sweeps, comms-stress curves, curve aggregation
├── export/        # ONNX export + typed sidecar; ONNX-Runtime equivalence gate; publish
├── track/         # experiment tracking (MLflow default) + provenance capture
└── bench/         # the reference-score harness Bench consumes
tests/learn/       # mirrors the package layout
```

## Extras

The base wheel is a lightweight env-adapter library; every heavy runtime lives behind an extra.

| Extra | Adds | Needed for |
|---|---|---|
| `rllib` | Ray RLlib + CPU Torch | training any baseline |
| `export` | `onnx`, `onnxruntime` | `PolicyPackage` export + the equivalence gate |
| `eval` | `pyarrow` | Parquet curve aggregation |
| `jax` | `jax` (CPU; install `jax[cuda12]` for GPU) | the batched GPU-vectorized rollout |
| `mlflow` | `mlflow` | the MLflow tracking backend / sink |

## Baselines

Four registered, reproducible plugins (`astro_mine.learn.algos`), discovered by capability tag
and replaceable by any third-party plugin through the `astro_mine.learn.algorithms` entry-point
group:

| Tag | Paradigm | Notes |
|---|---|---|
| `ippo` | independent | the simple control — no shared information |
| `mappo` | CTDE | decentralized actors, centralized critic over `SwarmEnv.state()` |
| `qmix` | CTDE | in-house QMIX/VDN over the discrete task selector |
| `comms_ppo` | CTDE + **comms-learning** | MAPPO plus a **differentiable message channel** |

`comms_ppo` is the [learn.md §11](https://github.com/astro-mine/docs/blob/main/architecture/learn.md)
comms-learning research track. Its learned messages ride the **same** `CommsModel` channel every
other baseline is scored on: an agent conditions on the mean-pool of the messages from the peers
the channel actually *delivered* this tick (after gate → budget → drop → delay), and the message
encoder is trained end-to-end by the team objective — so a comms-learning result stays
leaderboard-comparable to a comms-blind one.

## Training (tier 1 → distributed → GPU-vectorized)

One command trains a baseline on a single workstation (learn.md §7 tier 1). The world is a
`module:attr` factory, resolved at runtime — so the producing package is never a Learn dependency
and Learn never imports it:

```bash
uv sync --extra learn-rllib     # Ray RLlib + CPU Torch (the training path)
astro-mine learn --algorithm mappo \
    --env-factory astro_mine.sim.reference:make_reference_env_and_assets \
    --seed 0 --iterations 100 --hidden-sizes 64,64
```

`astro_mine.sim.reference` ships a small synthetic three-agent scenario as package data — no content store,
no registry, no network — which is what makes this command runnable as written. Swap in your own
factory whenever you have one; `--env-factory` is a **replaceable seam, not a privileged path**.

A factory may yield **either** shape:

| Returns | For |
|---|---|
| a `SwarmEnv` | a producer that already depends on Learn |
| an `(Environment, {AgentId: Asset})` pair of **Core** types | a producer that must not — Learn wraps it |

The pair form is the reason a simulator can supply a world without either package importing the
other (`conventions.md` §1.1): a Core `Environment` alone is not enough, because per-agent
observation and action spaces are derived from the SADF, so both halves have to cross.

**Reward shaping is yours.** A Core `Environment` leaves `StepResult.rewards` empty — a simulator
renders physics, not training signal — so `default_reward_fn` (or a function you pass) produces
the return. What matters is that your env's *actions move the agent*: against a world whose
dynamics ignore actions, any pose-derived reward is flat with respect to the policy and training
is vacuous.

A reference `TrainConfig` ships too, as a starting point to copy and edit:

```bash
uv run python -c "from astro_mine.learn.reference import train_config_path; print(train_config_path())"
# → pass that path to --config-json
```

The **same command** scales out unchanged — "the same code with a different executor, never a
fork" (learn.md §2.1). The rollout executor is selected from the fidelity axis:

- `--num-workers N` (or a Ray `--ray-address`) → the **KubeRay** distributed executor;
  Cloud wraps this entrypoint in a `RayJob` and injects its `RunContext` provenance envelope
  (read from env vars into the produced-policy provenance — Learn never calls Cloud).
- `--fidelity gpu_vectorized --batched-world your_pkg:make_gpu_world` → the **batched**
  vector executor (below).
- `--fidelity surrogate` flags the exported policy for a high-fidelity validation pass.

Determinism is topology-fixed: a single-worker distributed/vector run is byte-identical to
the in-process run for the same seed.

### Exporting the trained policy

`--export` writes the trained policy as an ONNX **`PolicyPackage`** — the one artifact
Mind/Guard/Bench consume (RM-P1-LEARN-05) and the commons' unit of exchange:

```bash
uv sync --extra learn-rllib --extra learn-export
astro-mine learn --algorithm mappo \
    --env-factory astro_mine.sim.reference:make_reference_env_and_assets \
    --seed 0 --iterations 100 --hidden-sizes 64,64 \
    --export ./policies
```

The command comes from [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli), a separate
distribution that depends on this one; this package ships no console scripts. Learn's group is flat
— no subcommand — so the address is `astro-mine learn <flags>`, and there is no bare
`astro-mine train` or `astro-mine-learn`.

Each agent's actor becomes its own graph, written to a **content-addressed** store —
`./policies/<hex>/{model.onnx,policy_package.json}` — and the `sha256:` graph digest of each is
printed to stderr (the report JSON keeps stdout clean). That digest *is* the artifact identity:
carry it to `astro-mine hub publish` or a leaderboard submission.

Every graph passes the **ONNX-Runtime equivalence gate** before it is written — a graph that
diverges from its Torch source fails the command non-zero and leaves nothing on disk. The
sidecar carries honest provenance (seed, toolchain, lockfile, and Cloud's `RunContext` envelope
when present) plus the declared comms/observability assumptions and any surrogate-fidelity
caveats, so Guard knows the envelope to enforce. `--export-version` stamps the `PolicyPackage`
version (default `0.1.0`); `--export-format` accepts `onnx`, the only cross-component policy
artifact (learn.md §2.5).

### GPU-vectorized rollout

`VectorExecutor` runs a **genuinely batched kernel** — one world step and one policy forward per
agent per tick, for the *whole* batch of env copies — whenever it is given a `BatchedWorld` and
the trainer's step is a `BatchedStep` (every Learn baseline's is). `BatchedWorld` is the protocol
**Sim's Brax/MJX GPU tier plugs into**; Learn ships `JaxBatchedWorld` as a reference JAX/XLA
realization (a jit-compiled, vmapped program over a device-resident batch — *not* a physics
model).

Without a batched world, without the `[learn-jax]` extra, or on a workstation with no GPU, it degrades
**gracefully** to the sequential CPU loop — tier 1 must always work. `VectorExecutor.backend`
reports which path was actually taken.

**Measured throughput** (`python -m astro_mine.learn.envs.vector.benchmark --num-envs N
--env-factory …`; MAPPO's real rollout step, 32 ticks, 3 heterogeneous agents):

| `num_envs` | sequential CPU loop | batched kernel (XLA-CPU) | speedup |
|---:|---:|---:|---:|
| 8 | 401 env-steps/s | 3,576 env-steps/s | **8.9×** |
| 64 | 416 env-steps/s | 18,911 env-steps/s | **45.5×** |
| 256 | 468 env-steps/s | 26,234 env-steps/s | **56.0×** |

Measured on an **XLA-CPU** host (no accelerator present), so this speedup is from *batching
alone* — the accelerator multiplies it. The batched kernel is the same jit/vmap program on CPU
and GPU, which is why CI covers it on CPU and only the device-residency assertion is `gpu`-marked.
The `gpu`-marked test (`tests/envs/test_vector_batched.py`) re-runs this benchmark on a real
accelerator; run it with `uv run pytest -m gpu` on a CUDA host with `jax[cuda12]` installed.

## Curricula

learn.md §11's MVP: hand-authored staged curricula + domain randomization, with an
**automatic-curriculum plugin interface** (the Phase-2 deferral) open by construction.

```python
from astro_mine.learn.curriculum import comms_ladder, StagedCurriculum, run_curriculum

curriculum = StagedCurriculum(comms_ladder(), seed=0, base=config)
report, trainer = run_curriculum("mappo", curriculum, stage_env_factory, iterations=200)
```

`comms_ladder` walks the swarm up the charter §8 difficulty gradient (clear channel → lossy +
delayed → range-gated, bandwidth-starved); `randomized_comms` re-samples the channel every
episode so a policy cannot overfit to one drop rate. A promotion rebuilds the **world** while the
learner (nets, optimizer, seeded RNG) carries over. An automatic curriculum (PLR, teacher-student)
is just a `Curriculum` whose `update(metrics)` picks the next stage differently; it registers
through the `astro_mine.learn.curricula` entry-point group.

## Experiment tracking

`TrackedRun` captures the `TrainConfig`, the comms regime, the curriculum, the seeds, the
toolchain, and the lockfile hash, and **content-addresses** them into `run_hash` — the
reproducibility key Bench re-derives from. It streams the learning curve and the honest-eval
curves into that same run and links the produced policy's ONNX digests back to it.

```python
from astro_mine.learn.track import TrackedRun, MlflowBackend

with TrackedRun(config, algorithm="mappo", comms=comms_cfg, backend=MlflowBackend()) as run:
    for _ in range(config.iterations):
        run.log_iteration(trainer.train_iteration())
    run.log_export(trainer.export(), digests=digests)
```

MLflow is the default backend (`[learn-mlflow]` extra); with no backend it records into an
`InMemoryBackend`, so a tier-1 run still gets its full provenance record with no server and no
network.

## Development

Learn is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup. Tier 1 trains a baseline on a single
workstation before any cluster is involved.

```bash
uv sync --extra learn-rllib --extra learn-export \
        --extra learn-eval --extra learn-jax     # what CI installs
python scripts/test.py learn                     # the CI gate
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
