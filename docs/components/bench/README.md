# astro-mine-bench

**Benchmark suite, scenario zoo, and reproducibility harness for [Astro-Mine](https://github.com/astro-mine).**
Named challenge scenarios, standard metrics, public leaderboards, and a deterministic
reproducibility harness — *clone, run the anchor scenario, and score a baseline in an
afternoon*, offline, with no account. Reproducibility is the product.

> **Status:** Phase 1 — the flywheel. Phase 0 is complete and green: the anchor scenario, the
> reference metric set, the baseline policy, the reproducibility harness + determinism gate, the
> local scoring tier, and the leaderboard service all ship. Phase 1 is landing on top: Hub-digest
> submission intake (RM-P1-BENCH-10), scale-out evaluation on Cloud (RM-P1-BENCH-11), community
> metric plugins + the View leaderboard/replay dataset (RM-P1-BENCH-12), and the hosted tier's
> security and observability posture — OIDC + OPA + supply-chain verification, sandboxed submission
> execution, OpenTelemetry/Prometheus, and the Postgres/pgvector scenario catalog. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/bench.md) and the
> [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Layout

```
src/astro_mine/bench/       # import path: astro_mine.bench  (local scoring: astro_mine.bench.run)
  scenario/ zoo/ metrics/ harness/ baseline/ leaderboard/ submit/ eval/ recording/ report/ sandbox/
tests/bench/                # mirrors the package layout

# The leaderboard's Rego policy, Prometheus scrape config and Grafana dashboard ship with the
# REST tier that deploys them: `deploy/` in astro-mine-api (RM-DIST-03).
TRUST_BOUNDARY.md           # what the submission sandbox protects — and what it does not
```

`policy/`, `deploy/` and `TRUST_BOUNDARY.md` sit at the repo root: Bench resolves them by
root-relative path, so consolidation kept them where the code already looked for them.

## Usage

**Local scoring (offline, no account, no cloud)** — clone, run the anchor, score a baseline:

```bash
astro-mine bench score          # the scorecard
astro-mine bench score --json   # machine-readable
astro-mine bench list           # scenarios in the zoo
```

The command line is [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli), a separate
distribution that depends on this one; this package ships no console scripts. There is one address
for every command on the platform — `astro-mine <component> <verb>` — so `score` is reached as
`astro-mine bench score` and nowhere else.

`python -m astro_mine.bench` is **not** a second spelling of it. That module provides exactly one
machine-facing entry point, `eval-worker` — the per-seed rollout Cloud fans out — and anything else
exits 2 with a message saying so.

Or from Python — inject any Core `Policy` (the Sim-backed runner in `astro_mine.sim.bench` slots
into the same seam; Bench itself stays dependency-clean, `core + pydantic`):

```python
from astro_mine.bench.baseline import BaselinePolicy, run
from astro_mine.bench.zoo import load_scenario, ANCHOR_SCENARIO_ID

scorecard = run(load_scenario(ANCHOR_SCENARIO_ID), BaselinePolicy())
```

**Hosted leaderboard (submit-policy-we-run + held-out seeds + sampled re-execution)** — an
optional service tier; the local tier above needs none of it:

The FastAPI route module is **not** part of `astro-mine-platform` (`docs/CONSOLIDATION_PLAN.md`
§"Not migrated") — the REST tier is destined for `astro-mine-api`, which is not stood up yet
(roadmap `RM-DIST-03`), so there is nothing to install for it today. The leaderboard *library*
underneath it did come across, behind the `bench-leaderboard` extra, and `create_app()` raises a
message saying exactly this rather than failing on an import.

```bash
pip install -e '.[bench-leaderboard,bench-observability]'   # the library half
docker compose up --build                                   # Postgres/pgvector

# Reads are account-free — the board, scorecards, provenance, replays:
curl localhost:8000/leaderboard/lunar-polar-ice-prospecting-v1

# Writes require an OIDC bearer token (bench.md §9); an unauthenticated POST is rejected:
curl -X POST localhost:8000/submissions -H "Authorization: Bearer $TOKEN" \
  -d '{"scenario_id":"lunar-polar-ice-prospecting-v1","policy_ref":"astro_mine.bench.baseline:BaselinePolicy"}'
```

### Submitting from the CLI

`submit` is the write path — the last step of the flywheel, and the one that used to be `curl`
with a hand-assembled JSON body:

```bash
pip install -e '.[bench-submit]'
export ASTRO_MINE_BENCH_TOKEN=<oidc token>       # never a flag: a token on argv lands in `ps`

# The path a community submission should take: an artifact referenced by digest.
astro-mine bench submit --hub-ref sha256:… \
  --scenario-id lunar-polar-ice-prospecting-v1 --to https://board.example --wait
```

`--hub-ref` takes a Hub `name:version` tag or a `sha256:` digest. Bench resolves it from Hub and
verifies it fail-closed — content address, then cosign signature, SLSA provenance, and SBOM —
before running it sandboxed. This is the contract `bench.md` §6 fixes: *a leaderboard submission
references Hub artifacts by digest*, which is what keeps an entry reproducible.

`--policy-ref module:attr` also works and is the **local/dev** path. It runs sandboxed like any
submission, but nothing pins what the reference resolves to — a re-run can import different code
under the same name — so it is **not leaderboard-grade** and no architecture document describes it
as an intake. Prefer `--hub-ref` for anything meant to stand as a published result.

The Hub intake returns a **job ticket**, since evaluation is asynchronous. `--wait` polls it to a
terminal status and prints the resulting submission and its rank; without `--wait` the job id is
printed along with the command to resume:

```bash
astro-mine bench submit --job <job-id> --to https://board.example --wait
astro-mine bench submit … --json          # machine-readable, like `score --json`
```

Identity comes from the verified token and **only** from the token — no flag can set it, and the
request body carries no identity field (bench#29). Reading the board needs no account at all;
`score` and `list` are untouched by any of this and keep working offline with no token and no
extra installed.

The hosted tier runs every submitted policy **out-of-process in a sandbox** — no network egress,
hard CPU/memory/time caps — because a submission is untrusted code (bench.md §9). Read
[TRUST_BOUNDARY.md](../../../TRUST_BOUNDARY.md) before exposing a leaderboard to the public internet: it
states exactly what each sandbox tier does and does not protect against. Hub-digest submissions are
additionally verified for a **cosign signature, SLSA provenance, and an SBOM** before they execute,
reusing [Seal](https://github.com/astro-mine/astro-mine-seal)'s primitives via the Hub client —
verification failure fails closed.

Authorization is an OPA-shaped policy layer (RBAC + per-role submission quotas + embargo control):
`deploy/policy/bench.rego` in astro-mine-api is the Rego an OPA sidecar evaluates, and the
in-process `RbacPolicyEngine`
enforces the same rules when no sidecar is configured. Every authN/authZ decision, verification
outcome, and sandbox rejection lands in a queryable audit trail (`GET /audit`).

**Observability** (`--profile observability`): OpenTelemetry spans cover `submit → evaluate → score
→ rank` with the trace context propagated across the queue hop, and `GET /metrics` exposes the
Prometheus series — **queue depth**, the **re-execution mismatch rate** (bench.md §10's key
integrity signal), and **evaluation latency**. astro-mine-api's `deploy/grafana/` has the starter
dashboard.

**Scenario catalog:** the packaged zoo is scanned from the filesystem by default (offline, no
database). Setting `ASTRO_MINE_BENCH_CATALOG_DSN` switches discovery to the **Postgres + pgvector**
catalog (bench.md §5) — spec/version/lineage index plus similarity search:

```bash
astro-mine bench zoo-sync --dsn "$DSN"                  # seed it from the packaged zoo
astro-mine bench zoo-search --dsn "$DSN" "ice prospecting endurance"
```

## Development

Bench is part of the [`astro-mine-platform`](../../../README.md) distribution — one repository, one
environment, one test suite. See [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup, then run
this component's suite with its own CI selection:

```bash
python scripts/test.py bench
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full
workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
