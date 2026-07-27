# Astro-Mine-Cloud platform (Helm)

The Phase-1 Cloud MVP: **one `helm install`** stands up the curated scale-out substrate on any
conformant Kubernetes cluster (v1.29+) — [`cloud.md` §3, §4, §7, §12](https://github.com/astro-mine/docs/blob/main/architecture/cloud.md).
This directory is declarative infra (it is **not** part of the `astro-mine-cloud` Python wheel).

## What it installs

| Piece | Role | cloud.md |
|---|---|---|
| **KubeRay** | RayJob/RayCluster — tightly-coupled RL / actor fleets / solves | §2.3, §4 |
| **Argo Workflows** | DAG / fan-out batch sweeps | §2.3, §4 |
| **Kubernetes Jobs** | trivial one-shots (native, no operator) | §2.3 |
| **NVIDIA GPU Operator** | MIG partitioning + time-slicing + DCGM telemetry | §4, §8 |
| **Kueue** | queueing, quotas, fair-share | §4 |
| **kube-prometheus-stack + Loki** | metrics/dashboards + logs | §10 |
| **Kyverno** + its `ClusterPolicy` (ours) | cosign-verified-images-only admission | §9 |

The three engines are what the `astro-mine-cloud` submission library compiles onto
(`astro_mine.cloud.engines`): a `JobSpec` routes to Ray or a K8s Job, a `SweepSpec`/`WorkflowSpec`
to Argo. Backend equivalence holds — the *same* `submit()` call site runs local↔cluster.

## Install

```sh
helm dependency build platform/helm/astro-mine-cloud       # resolve the operator subcharts
# a laptop kind/k3s cluster (engines only, fast):
helm install astro-mine platform/helm/astro-mine-cloud -f platform/profiles/kind.yaml
# a production cluster (full stack + GPU + admission):
helm install astro-mine platform/helm/astro-mine-cloud -f platform/profiles/prod.yaml
```

Lifecycle-manage via **GitOps** (Argo CD / Flux) for reproducible, auditable cluster state.

## Profiles

- **`profiles/kind.yaml`** — engines + Kueue only; GPU/observability/admission off. Exercises the
  *same* charts on a laptop kind cluster. Distinct from the dependency-free local tier, which
  needs no Kubernetes at all.
- **`profiles/kind-admission.yaml`** — the kind profile **plus** Kyverno and the cosign policy,
  scoped to one namespace and one image prefix. The scoping is not cosmetic: an unscoped
  `imageReferences: ["*"]` rule refuses the platform's own (unsigned, upstream) operator pods —
  Kyverno's included — and takes the cluster down with it.
- **`profiles/prod.yaml`** — the full stack, GPU Operator, observability, and cosign admission.

## The live-cluster harness (`platform/kind/`)

`helm lint` renders templates. It does not tell you whether the operators come up, whether a
compiled manifest is one the API server accepts, or whether a cluster run reproduces a laptop run.
The harness here does — it stands up a real ephemeral cluster and runs the `cluster`-marked tests
against it ([`cloud.md` §10](https://github.com/astro-mine/docs/blob/main/architecture/cloud.md)).

```sh
./platform/kind/up.sh                        # kind + registry + MinIO + a real helm install
set -a && . ./platform/kind/harness.env && set +a
uv run pytest -m cluster
./platform/kind/down.sh
```

Add `ADMISSION=1` to `up.sh` to also install **Kyverno** and the cosign policy
(`profiles/kind-admission.yaml`), which enables the supply-chain admission tests. Needs `docker`,
`kind`, `kubectl`, `helm` (and `cosign` + `opa` for the admission/authz tests).

| Piece | Why |
|---|---|
| `cluster.yaml` | 3 nodes — **two workers**, so the chaos test can take one away and the retry still has somewhere to land |
| `registry.sh` | a registry the *cluster* can reach: Kyverno fetches signatures from inside its own pod, where `localhost` is the Kyverno pod |
| `minio.yaml` | the shared object store — one bucket, reached by pods via cluster DNS and by the host via a node port |
| `workload.Dockerfile` | the workload image; its entrypoint is the **in-pod run harness** |
| `up.sh` / `down.sh` | idempotent setup / teardown (`down.sh` is safe to run when nothing is up, so CI can call it from `always()`) |

**Two pins in the image build are load-bearing.** `RunContext.code_version` and `env_lockfile` are
both *inside* the content address the determinism gate compares
([`conventions.md` §5](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)),
so the image is built with `SETUPTOOLS_SCM_PRETEND_VERSION` set to the host's version and ships
the repo's `uv.lock` at its `WORKDIR`. Get either wrong and the equivalence test fails for reasons
that have nothing to do with the run. The fix is never to drop fields from `content_address()`.

## CI

- **`ci.yml`** (every push/PR) runs `helm lint` against the chart with each profile's values, plus
  the hermetic Python suite. Fast, no Docker, no cluster.
- **`cluster-e2e.yml`** (opt-in) runs the harness above: `helm dependency build` + `helm install`
  on a real kind cluster, dispatch round-trips, the determinism gate, chaos, Kueue back-pressure
  and admission. It runs on `workflow_dispatch`, nightly, and on a PR labelled **`cluster-e2e`** —
  never by default, because it takes tens of minutes and needs a Docker daemon.
