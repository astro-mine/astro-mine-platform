# astro-mine-cloud

**Distributed orchestration for [Astro-Mine](https://github.com/astro-mine).**
The Phase-0 *discipline* — container-first packaging, a `submit()` backend-equivalence
contract, and content-addressed artifact I/O with a provenance envelope — grows in Phase 1
into the **scale-out substrate**: typed job/sweep/workflow contracts that compile to
Ray/Argo/K8s, a Helm-installable platform, Kueue scheduling with budgets, a data-locality
layer, and MLflow + namespace-per-tenant isolation + cosign admission. **Local-first, never a
hard dependency:** the same call site runs on a laptop or a cluster, and every cluster piece is
a pure, unit-testable manifest builder (heavy runtimes ride optional extras).

> **Status:** Phase 1 — the Cloud MVP (RM-P1-CLOUD-01…06). The contracts, engine compilers,
> scheduling/budget/checkpoint logic, data-locality layer, tenancy/admission policy, and MLflow
> wiring ship as tested Python; the platform ships as Helm charts (`platform/`). Live-cluster
> execution and real MLflow are opt-in behind extras. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/cloud.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Layout

```
src/astro_mine/cloud/       # import path: astro_mine.cloud
  artifacts/ packaging/     # content-addressed I/O + RunContext; digest-pinned images  (Phase 0)
  submission/               # JobSpec/SweepSpec/WorkflowSpec, submit(), cluster backend, CLI
  engines/                  # compile a spec -> K8s Job / KubeRay RayJob / Argo Workflow
  sched/ autoscale/ gpu/    # Kueue + budgets; spot + checkpoint-resume; MIG/DCGM
  data/                     # lazy Zarr/COG/Parquet chunk-streaming + pull-through cache
  runs/ tenancy/            # MLflow + events; namespace isolation + cosign admission
  k8s/                      # shared manifest helpers (labels, naming, YAML)
platform/                   # Helm charts + kind/prod profiles (declarative infra, not the wheel)
tests/cloud/                # mirrors the package layout
```

## Artifact I/O + provenance (`astro_mine.cloud.artifacts`)

Content-addressed storage plus a `RunContext` envelope, so a laptop run and a future
scaled run start from identical bytes. The default store is dependency-free and keeps
the local tier sacred (no cloud, no account); the same call site swaps to an
S3-compatible backend without a code change.

```python
from astro_mine.cloud.artifacts import FilesystemArtifactStore, RunContext, content_address

store = FilesystemArtifactStore()                 # or S3ArtifactStore(bucket=...)  [s3 extra]
address = store.put(b"...bytes...")               # -> "sha256:<hex>"
assert store.get(address) == b"...bytes..."

ctx = RunContext(
    source_content_hashes={"scenario": content_address(spec)},  # inputs by hash
    env_lockfile=content_address(lockfile_bytes),
    seed=42,
)
ctx.store(store)                                   # provenance is itself an artifact
```

The S3 backend is optional — `pip install 'astro-mine-platform[cloud-s3]'`. The opt-in
integration test needs a MinIO on `$MINIO_ENDPOINT`:
`MINIO_ENDPOINT=http://localhost:9000 uv run pytest -m minio`. Cloud's per-repo
`docker-compose.yml` did not come across the consolidation, so bring one up yourself.

## Container packaging (`astro_mine.cloud.packaging`)

Every Phase-0 workload is packaged as a **digest-pinned** OCI image built **reproducibly**
on a **pinned base** — so a laptop image and a future cluster image are byte-identical, and
no cluster is required to build one. `ImageRef` is the digest-pinned reference every run
consumes; `render_dockerfile` emits the reproducible recipe (pinned base, fixed
`SOURCE_DATE_EPOCH`, no build-time network, non-root); `build_image` returns the pinned
result.

```python
from astro_mine.cloud.packaging import BuildSpec, ImageRef, build_image, render_dockerfile

spec = BuildSpec(
    base=ImageRef.parse("docker.io/library/python@sha256:..."),  # base pinned by digest
    repository="ghcr.io/astro-mine/astro-mine-sim",
    version="0.1.0",
    entrypoint=["python", "-m", "astro_mine.sim"],
)
dockerfile = render_dockerfile(spec)          # deterministic; see templates/workload.Dockerfile
image = build_image(spec)                      # -> ImageRef pinned by the built digest
```

`build_image` shells out to `docker buildx` only when actually building; `render_dockerfile`
and the rest are pure. An unpinned reference (`repo:tag` with no `@sha256:…`) is rejected at
the boundary, so an unpinned image can never enter a job.

## `submit()` backend-equivalence contract (`astro_mine.cloud.submission`)

One call site runs a workload the same way on a workstation as it later will on a cluster —
a **backend swap, not a code fork**. Phase 0 registers two dependency-free local backends:
`local` (a subprocess in your Python env) and `docker` (the same job in its digest-pinned
image via `docker run`). The same `JobSpec` through either yields **identical
content-addressed outputs and provenance** — that equivalence is the contract.

```python
from astro_mine.cloud import submit
from astro_mine.cloud.artifacts import FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec

store = FilesystemArtifactStore()                              # local, no account
job = JobSpec(
    image=ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:..."),
    command=["python", "-m", "astro_mine.bench"],
    inputs={"scenario.json": store.put(b"...")},              # staged by content address
    outputs=["score.json"],                                    # captured back by content address
    seed=42,
)
result = submit(job, store=store)              # backend="local" (default) | "docker"
assert result.ok
```

The workload reads inputs from `$ASTRO_MINE_INPUTS`, writes outputs to `$ASTRO_MINE_OUTPUTS`,
and sees the seed in `$ASTRO_MINE_SEED`. Every run records a `RunContext` (image digest,
seed, input hashes, outputs) in the store. The opt-in real-Docker test runs with
`ASTRO_MINE_DOCKER_IMAGE=<image@digest> uv run pytest -m docker`.

## Phase 1 — the scale-out substrate

### Sweeps, workflows, engines, CLI (`astro_mine.cloud.submission`, `.engines`)

A `SweepSpec` (grid / random / low-discrepancy) and a `WorkflowSpec` (a validated DAG) sit on
top of `JobSpec`. Each compiles to the right engine for its shape — a `JobSpec` to a **KubeRay
RayJob** (tightly-coupled) or a plain **K8s Job** (one-shot); a sweep/workflow to an **Argo
Workflow** — as pure manifest dicts, no cluster needed to build or test one. The `cluster`
backend keeps the equivalence contract: the *same* `submit()` call runs local↔cluster.

```python
from astro_mine.cloud.engines import compile_sweep, select_engine
from astro_mine.cloud.submission import JobSpec, SweepSpec

sweep = SweepSpec(base=job, grid={"lr": [0.1, 0.2], "bs": [16, 32]}, max_parallel=8)
workflow = compile_sweep(sweep, namespace="acme")   # -> an Argo Workflow (4 fan-out tasks)
```

[`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli) drives it from the shell (this
package ships no console scripts):

```bash
astro-mine cloud submit job.json --backend cluster --input scenario.json=./scenario.json
astro-mine cloud expand sweep.json          # preview the expansion
astro-mine cloud compile job.json           # the engine manifest it would run as
```

The same handlers were also reachable over **REST** — a thin FastAPI edge delegating to the
identical `submit()`/engine paths. `astro_mine.cloud.serve` is **not** part of this distribution
(`docs/CONSOLIDATION_PLAN.md` §"Not migrated") and its `[serve]` extra was dropped with it; the
REST tier is destined for `astro-mine-api`, which is not stood up yet (roadmap `RM-DIST-03`).
Manifest generation, the contracts, scheduling and the data layer are all reachable from Python
without it.

### Scheduling, cost, resilience (`.sched`, `.autoscale`, `.gpu`)

Kueue `ClusterQueue`/`LocalQueue` quotas with fair-share admission; **hard per-tenant budget
caps** that halt a runaway sweep before it overspends; spot-first Karpenter node pools with
scale-to-zero; **content-addressed checkpoint-to-resume** so a preemption loses ≤ one interval
and the resumed run reproduces the uninterrupted result; MIG profiles + DCGM for GPU sharing.

### Data locality (`.data`)

Lazy **chunk-range** reads of Zarr / COG / Parquet from any S3-compatible store (`[cloud-s3]`), plus a
**pull-through cache** so a sweep's repeated reads are served from local scratch — never a bulk
copy — and node-affinity hints to co-schedule onto cache-warm nodes.

### Reproducibility, tenancy, trust (`.runs`, `.tenancy`)

Every job is an **MLflow** run recording its `RunContext` + content-addressed artifacts (`[cloud-mlflow]`);
completion events go on NATS for Bench/Studio/Hub. The opt-in test needs a tracking server on
`$MLFLOW_TRACKING_URI`: `MLFLOW_TRACKING_URI=http://localhost:5000 uv run pytest -m mlflow`. `tenant_manifests()` builds the
namespace-per-tenant baseline (RBAC, quotas, **default-deny NetworkPolicies**); `admit()` and the
shipped Kyverno policy enforce **cosign-verified-images-only** admission (signed + SLSA + SBOM +
compatible Core version), refusing anything else at the cluster boundary.

### The platform (`platform/`) — and a cluster that proves it

One `helm install` stands up KubeRay + Argo + Kueue + Kyverno + the GPU Operator + observability,
with `kind`, `kind-admission` and `prod` profiles.

The cluster claims are not taken on faith. `platform/kind/` scripts an **ephemeral kind cluster**
that the opt-in `cluster`-marked tests run against: a real `helm dependency build && helm install`,
a real K8s Job and a real RayJob dispatched through the *same* `submit()` a laptop uses, pods and
nodes killed mid-run to prove checkpoint-resume recovers, Kueue holding a tenant at quota, and a
live Kyverno refusing an unsigned image. The centrepiece is the **determinism gate**: the cluster
run's `RunContext` content-address must equal the local run's — same job, same bytes, different
substrate.

```bash
./platform/kind/up.sh                        # ADMISSION=1 to also install Kyverno
set -a && . ./platform/kind/harness.env && set +a
uv run pytest -m cluster
./platform/kind/down.sh
```

It runs in CI as the opt-in `cluster-e2e` workflow (nightly, on demand, or on a PR labelled
`cluster-e2e`) — never in the default suite, which stays hermetic. See
[`platform/README.md`](../../../platform/README.md).

### Optional extras

Heavy runtimes stay out of the sacred local tier: `pip install 'astro-mine-platform[cloud-s3]'`
(object I/O), `[cloud-cluster]` (pyyaml → render/apply manifests), `[cloud-mlflow]` (MLflow
tracking), `[cloud-nats]` (completion events). Manifest *generation*, contracts, scheduling, and
data logic need none of them. `[serve]` is gone rather than renamed — see above.

## Development

Cloud is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup.

```bash
python scripts/test.py cloud
```

The default suite is hermetic — no Docker, no cluster, no account. The integration tests that need
real infrastructure are **opt-in markers** and skip by default:

```bash
MINIO_ENDPOINT=http://localhost:9000       uv run pytest -m minio
NATS_URL=nats://localhost:4222             uv run pytest -m nats
MLFLOW_TRACKING_URI=http://localhost:5000  uv run pytest -m mlflow
ASTRO_MINE_DOCKER_IMAGE=<image@digest>     uv run pytest -m docker
./platform/kind/up.sh && set -a && . ./platform/kind/harness.env && set +a
                                           uv run pytest -m cluster
```

Each needs the named service already running; the per-repo `docker-compose.yml` that used to start
MinIO/NATS/MLflow did not come across the consolidation.

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
