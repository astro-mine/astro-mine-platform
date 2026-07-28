# Astro-Mine-Platform — consolidation plan

**Goal.** Consolidate all Python code from the 18 `astro-mine-*` component repos
into this single repository / single distribution (`astro-mine-platform`, package
`astro_mine`) with **100% functional equivalence**. This is a mechanical
consolidation, not a rewrite: files are copied byte-for-byte; code changes are
limited to import/path fixes, dependency merging, namespace/packaging
adjustments, and removal of cross-repo duplication. Every deliberate edit is
listed in this document (§6) and encoded in `scripts/consolidate.py`.

**Scope.**
- Migrated: every `astro_mine.<comp>` Python package, every test suite, every
  CLI (console scripts, the `astro-mine` umbrella, deprecated aliases), all
  entry-point groups, all schemas/examples/reference data, guard's Rust safety
  core (`rust/`, PyO3 → `astro_mine.guard._core`).
- **Not migrated (mandated):** REST API server code — `astro_mine.hub.api`,
  `astro_mine.studio.api`, `astro_mine.cloud.serve`, and
  `astro_mine.bench.leaderboard._app` (the FastAPI route module *only*; the
  leaderboard *library* — `_service`, `_sql`, `_auth`, `_authz`, `_eval`, … —
  is not REST and is migrated). gRPC services (`sim.service`,
  `prospect.service`) are not REST and are migrated.
- **Not migrated:** TypeScript/React UIs (`hub/ui`, `studio/ui`, `bench/ui`,
  and the `view`/`console` repos, which contain no Python).

## 1. Why this is (mostly) mechanical

Every repo already ships an implicit-namespace package `astro_mine.<comp>`
under `src/astro_mine/`, with zero module-path overlap, zero console-script
name collisions, and zero entry-point collisions across all 18 repos (verified
by `scripts/consolidate.py --analyze-only`). Cross-component coupling is
already mediated by entry points and lazy imports. Consolidation is therefore:
copy the 18 subtrees side by side, merge 18 pyprojects into one, and fix the
small set of places that assumed "my repo root" or "my distribution name".

## 2. Repository layout

```
src/astro_mine/<comp>/     the 18 component packages (copied, REST modules excluded)
tests/<comp>/              each repo's tests/ (one level deeper than before)
rust/                      guard's Rust crate (build.rs reads ../schemas/proto — see §3)
schemas/proto, schemas/json  core's schema sources (path- and digest-coupled: kept verbatim)
schemas/<comp>/            other owners' codegen-only proto sources (+ their buf configs)
examples/                  core's + guard's examples/ merged flat (no subdir collisions)
validator/, codegen/, dist/schema-bundle/   core's (digest/parity-test-coupled, verbatim)
embargo/, policy/, deploy/, TRUST_BOUNDARY.md, Dockerfile   bench's (path-coupled, verbatim)
docker/                    sim's (render_dockerfile-coupled, verbatim)
platform/                  cloud's k8s/kind/helm assets (test-coupled, verbatim)
validation/                worlds' reference oracles (test-coupled, verbatim)
benchmarks/                allocate's scale-benchmark output dir (test-coupled, verbatim)
anchor-signing.pub         hub's (verbatim)
docs/<comp>/               guard's and hub's docs/
scripts/                   test-coupled scripts flat (original names); scripts/<comp>/ for the rest
```

The root-level rule: any repo-root directory that `src/` code, a script, or a
committed digest resolves by a **root-relative path** is kept at the platform
root under its original name (all such dirs have exactly one owner — verified).
This makes `bench/leaderboard/_eval.py`'s `parents[4]/"embargo"` and every
script's `ROOT = parents[1]` resolve correctly **with zero source edits**.
Tests move exactly one level deeper (`<repo>/tests/x.py` →
`tests/<comp>/x.py`), so the only systematic test edit is `parents[N]` →
`parents[N+1]` in the ~15 files that resolved their repo root (§6.4).

## 3. Build system: maturin

Guard is maturin-built (PyO3 extension `astro_mine.guard._core`; its wheel
bundles the Rust trusted core — a safety requirement, not packaging trivia).
A wheel can have only one build backend, so **the platform adopts guard's**:
maturin with `python-source = "src"`, `manifest-path = "rust/Cargo.toml"`,
`module-name = "astro_mine.guard._core"`, `features = ["python"]`. The 17
hatchling repos carry no hatch-specific build logic beyond file inclusion,
which maturin reproduces (explicit `include` entries carry worlds' 4
force-included files, surrogate's `dem_excavation_v1.npz`, and hub's
`policy/rego/*` — see pyproject). `rust/build.rs` resolves
`../schemas/proto/...`; guard's `schemas/proto` is vendored at
`schemas/guard/proto`, so the crate keeps a copy of the 4 protos it compiles
(see §6.8). `[profile.release] panic="abort", lto=true, codegen-units=1` in
the crate is a safety requirement and is untouched. Building from source now
needs a Rust toolchain; CI installs it.

**Version.** The per-repo hatch-vcs versions are replaced by a static
`version = "0.1.0"`. Only core (`v0.4.0`) and spice (`v0.1.2`) ever tagged;
the other 16 repos' built dists already reported `0.0.0`.

## 4. Dependencies

One `[project.dependencies]` = the union of all 18 base dependency lists,
specs verbatim; where two repos disagreed the intersection is kept
(`numpy>=2.5.1`, `scipy>=1.18.0` — studio's floors). Internal `astro-mine-*`
pins (and the whole `[tool.uv.sources]` git-pin matrix, core `v0.3.0` tag +
sibling revs) disappear: components are in-package, which also retires uv's
single-git-source flag-day constraint. `fastapi` is dropped (its only base-dep
user was studio, exclusively for the not-migrated `studio.api`).

Tight pins preserved as-is: `gymnasium==1.2.2` (Ray exact-pin),
`ortools>=9.10,<9.11`, `protobuf>=7.35,<8`, `zarr>=3.0`, torch via the
`pytorch-cpu` explicit index, learn's `rllib`×`mlflow` uv conflict (now
`learn-rllib`×`learn-mlflow`).

**Extras** are component-prefixed (`<comp>-<extra>`) because bare names
collided (three `recording`s, three `cloud`s, …). Extras whose only content
was a sibling astro-mine package become empty markers (the sibling is now
always installed) — kept so documented install lines still resolve. Dropped
extras: `hub[service]`, `cloud[serve]`, `studio[serve]` (REST-only);
`bench[leaderboard]` is kept as `bench-leaderboard` minus fastapi/uvicorn
(the library half still needs sqlalchemy/psycopg/pyjwt/httpx).

> **SUPERSEDED (astro-mine-platform#1 / astro-mine-cli#12).** This section describes what the
> consolidation did, and it was accurate then. It is no longer true of the platform: the CLI
> surface has since been removed entirely. `[project.scripts]` is empty, the four
> `astro_mine.cli*` entry-point groups are no longer declared here, and every command lives in
> `astro-mine-cli`, which depends on this package. The four *plugin* groups below are unchanged.
>
> Left as written rather than edited, because this document is the record of the migration —
> rewriting it would erase what was actually done in commit `fd91454`.

## 5. CLI & plugin surface (RFC-0011) — preserved exactly

All 20 console scripts (16 prefixed binaries + `astro-mine` + the one-cycle
deprecated aliases `fleet`/`worlds`/`link`/`prospect`/`astro-mine-train`) and
all 8 entry-point groups are declared verbatim in the merged pyproject. The
umbrella's discovery (`entry_points(group="astro_mine.cli")`, lazy
`ep.load()`) is distribution-name-agnostic and needs no change; third-party
packages still extend the platform by registering into the same groups.
Cosmetic delta: `--help` provider attribution now reads
`astro-mine-platform 0.1.0 (…)`. The umbrella's static first-party
verb→distribution manifest (`_manifest.py`, used only to print install hints
when a first-party verb is *absent*) becomes dead code — every first-party
verb is always present in one distribution — and is left unmodified.

## 6. Every place code is edited (the complete rewrite list)

Everything not listed here is copied byte-for-byte.

### 6.1 REST exclusions and their seams
- `astro_mine/bench/leaderboard/__init__.py` — `create_app()` (a thin lazy
  wrapper over the excluded `_app`) now raises a clear "REST surface not
  included in astro-mine-platform" ImportError instead of importing `_app`.
- `astro_mine/studio/cli.py` — `serve` fails fast with the same clear message
  (its `build_serve_app()` imported the excluded `studio.api`); every other
  studio CLI verb is untouched.
- `astro_mine/cloud/serve/`, `astro_mine/hub/api/`, `astro_mine/studio/api/`
  — excluded wholesale; nothing else imported them (verified: `hub.client`,
  `hub/__init__`, `studio/__init__`, `cloud/__init__` are api-free).
- Excluded REST tests: bench `test_leaderboard{,_hosted,_security}.py`,
  `test_submit.py`; hub `test_api.py`, `test_admission.py`,
  `test_artifact_kind.py`; studio `test_api.py`, `test_api_catalog.py`,
  `test_api_hub.py`, `test_seed_fixture.py`, `test_ui_schema.py` (UI-coupled);
  cloud `test_serve.py`. Mixed files are split, keeping the non-REST tests:
  bench `test_report_view.py`, `test_telemetry.py`; studio `test_cli.py`.
  Non-REST coverage lost with these files is restored by characterization
  tests (§8).
- bench's ruff `flake8-bugbear extend-immutable-calls` fastapi entries dropped.

### 6.2 Distribution-name couplings (`importlib.metadata`)
Every `version("astro-mine-<comp>")` call site (29 across the tree, rewritten
mechanically by `consolidate.py`) now queries `"astro-mine-platform"` — a
one-token change preserving all surrounding logic and fallbacks; the seven
previously-unguarded provenance sites — worlds ×6, prospect
`priors/ingest.py` — would otherwise raise at runtime. Sites: the 18
component `__init__` / `_version` modules, plus `guard/audit/stream.py`,
`learn/algos/_torch_common.py`, `allocate/api/planner.py`,
`cloud/artifacts/runcontext.py` (generic helper — unchanged, callers pass the
dist name), `studio/provenance.py`. **Consequence, called out:** version
strings stamped into artifacts (worlds bundle provenance, bench toolchain
digests, link cache digests, guard MCAP streams, fleet/prospect manifests) now
read the platform version where a per-repo version (usually `0.0.0`) appeared
before — content-hash drift equivalent to any version bump, same as the known
Core-bump digest-drift behavior.
- `worlds/illumination/_registry.py` — the built-in field-model
  ownership check ("hard-error if a *foreign distribution* claims a built-in
  id") learns that `astro-mine-platform` is self, not foreign.

### 6.3 Test-package imports
`tests/` becomes a package (`tests/__init__.py` + per-component
`tests/<comp>/__init__.py`). The six repos using absolute `from tests.x
import …` (cloud, mind, learn, allocate, guard, surrogate) are mechanically
rewritten to `from tests.<comp>.x import …`. hub/studio's relative imports and
cli's `sys.path.insert(__file__-relative)` conftest work unchanged.

### 6.4 Repo-root path anchors in tests (`parents[N]` → `parents[N+1]`)
core `test_cli.py`, `test_schema_bundle.py`, `test_schema_digest.py`,
`test_validator_rust_parity.py`, the `examples/`-globbing suites; sim
`test_packaging.py`, `test_service.py`, `test_bench_runner.py`; worlds
`test_gravity.py`, `test_validation_psr.py`; bench `test_zoo_anchor.py`,
`_policies_hostile.py`, `test_determinism_gate.py`, `_factories.py`,
`test_eval.py`, `test_sandbox.py`, `test_telemetry.py`, `test_contracts.py`;
cloud `test_platform_charts.py`, `tests/cluster/conftest.py`; allocate
`test_schema.py`, `test_scale_benchmark.py`; surrogate `test_schema.py`;
guard `test_schema_compat.py`, `test_cli.py`; learn `test_env_factory_seam.py`;
mind `test_cli.py`; cli `test_new.py`, `test_installed_provider.py`. The
referenced root dirs keep their exact names/locations (§2), so the bump is the
whole fix.

### 6.5 Wheel-building tests (core, mind, guard, cli)
`uv build --wheel` at the (now platform) root builds the whole platform wheel
via maturin — slower (Rust compile) but the assertions on wheel contents and
`entry_points.txt` lines still hold and are kept. Adjusted assertions only
where the old claim is structurally per-repo:
- cli `test_packaging.py::test_declares_no_runtime_dependencies` and
  `test_installed_provider.py`'s exact-venv-package-set / "pulls in nothing"
  assertions — the umbrella is no longer a zero-dependency distribution; these
  assertions are replaced by "the platform wheel + provider fixture install
  and the umbrella drives third-party verbs" (all other assertions —
  laziness, exit codes, passthrough, scaffold round-trip, `python -m` parity,
  malformed-provider diagnostics — kept verbatim).
- guard/mind/core wheel-inventory assertions: unchanged (their files are still
  in the platform wheel; guard's "no examples/ in the wheel" still holds).

### 6.6 Umbrella/metadata self-checks
Each repo's `test_umbrella.py` / `test_deprecated_alias.py` asserts its
declared entry points exist in the installed metadata — satisfied by the
merged pyproject; any assertion on the *providing distribution's name* is
updated to `astro-mine-platform`.

### 6.7 Optional-sibling seams that become unconditional
`allocate/mind.py` and `guard/mind/plugin.py` (eager `astro_mine.mind`
imports reachable only via entry points), plus all lazy `astro_mine.hub`/
`sim`/`bench`/`cloud` imports behind extras: **no code change** — but the
"sibling absent" ImportError branches become unreachable in-platform. Tests
that assert absence-degradation by uninstalling a sibling are re-scoped or
skipped with a documented reason; learn's `test_base_install` import-blocker
still passes (it blocks modules, not distributions).

### 6.8 Guard crate proto path
`rust/build.rs` compiled `../schemas/proto/**` from the guard repo root.
Guard's 4 protos live at `schemas/guard/proto` (core owns root
`schemas/proto`), so the crate gets the vendored relative path updated in
`build.rs` — the one Rust-side edit.

### 6.9 Scripts
Test-coupled scripts stay flat in `scripts/` (original names, `ROOT =
parents[1]` still the repo root); all other maintainer scripts move to
`scripts/<comp>/` and any `parents[1]`-style root anchor in them gets the same
+1 bump. Colliding names across repos (`check_model_drift.py`,
`export_schemas.py`, `gen_proto.sh`) exist only in the per-component tier.

## 7. Testing & CI

- Markers: union of all repos' registrations (same names already meant
  compatible things; `cluster`/`integration` variants documented in the
  registration text). Global `addopts = "-ra"`.
- Each repo's *default selection* differed (`sim` deselected gpu/ray/docker in
  addopts; allocate deselected `scale`). A single rootdir has one addopts, so
  `scripts/test.py` re-applies each component's original default when running
  that component — CI runs per-component jobs exactly mirroring each source
  repo's CI selection (`mind`: `-m "not pddl and not native and not slow and
  not sim"`, `learn`: `-m "not slow and not gpu and not cluster"` with rllib
  installed, `guard`: `-m "not sim and not slow"`, `hub`: `-m "not
  integration"`, dedicated `sim-e2e`/`cluster`/`scale` opt-in jobs, etc.).
- Coverage: `source = src/astro_mine`, omit `*/_proto/*` (generated),
  `*/__main__.py`, `*/mind/native/*` (per-repo omits, union), shared
  `exclude_also` block; target ≥95% line coverage, matching the per-repo
  `--cov-fail-under=95` gates.

## 8. Characterization tests

Where a migrated module lost its test file to a REST exclusion but is itself
non-REST (bench `report`/`telemetry` halves of the split files, the
`bench.submit` httpx client, `bench.leaderboard._service` & friends, studio
CLI non-serve paths), characterization tests are added under
`tests/<comp>/test_characterization_*.py` that pin current behavior
(inputs → outputs, error messages, file formats) without inventing new
behavior. Same for any coverage gap below the 95% gate.

## 9. Documented convention overrides

The docs mandate per-repo packaging in several places; this consolidation
consciously overrides them (per the platform mandate) while preserving the
*intent* where it matters:
- `conventions.md §7` multi-repo packaging; `VERSIONING.md §2` per-repo
  SemVer/tag series → one repo, one version series.
- `conventions.md §13` dist-name rule `astro-mine-<name>` per component → one
  dist; **import paths unchanged**.
- RFC-0011 "components stay independently installable" / CX-LOCAL; rejected
  alternative (a) → answered with component-prefixed extras + the existing
  lazy-import discipline (base install stays as light as the union of base
  deps allows).
- `conventions.md §1.1` "no component imports another" → in-package direct
  imports now *allowed*, but not introduced anywhere by the migration; all
  existing entry-point seams stay functional for third parties.
- RFC-0009's downstream canary (Core CI vs consumers@HEAD) → superseded: every
  consumer's schema tests run in-repo on every change, satisfying the intent
  trivially.
- Schema `$id`s, `SCHEMA_DIGEST`, `reference/` package-data paths: **not**
  overridden — preserved byte-for-byte (they are load-bearing in
  content-addressed artifacts).

## 10. Execution log — deltas found while running the migration

Beyond §6, execution surfaced these additional per-repo couplings (each fixed
in the same spirit; all changes live in this repo's git history):

- **String dotted-path refs** escaped the import rewriter: bench metric-plugin
  and policy entrypoint strings (`"tests._factories:…"`), learn env-factory
  refs (`"tests.fakes:…"`) → qualified with the component test package.
- **allocate** used `from tests import constraint_factories` (bare form).
- **worlds** `illumination/_registry.py` `_SELF_DISTRIBUTION` and its tests'
  fake-dist parametrization → `astro-mine-platform` (§6.2's last bullet, as
  predicted).
- **learn** `track/run.py` toolchain probe list (`astro-mine-learn`/`-core` →
  `astro-mine-platform`); its test expectation updated to one platform entry.
- **core** wheel tests: maturin writes `name=target` without spaces in
  `entry_points.txt` (normalized, guard's idiom); the "no .proto in the wheel"
  claim scoped to `astro_mine/core/` (sim always shipped its service proto).
- **cli**: the venv-integration fixture pins `uv venv --python sys.executable`
  (the platform wheel is cp312-binary, the old pure wheel was not); the
  provider fixture path became test-file-relative; the two zero-dependency
  assertions re-scoped per §6.5; manifest-distinction tests use
  `astro-mine-platform` as the guaranteed-installed dist; the plugin-kind
  listing expectation reflects `solver` being registered rather than
  "not installed".
- **bench**: README status-line test re-pointed at `docs/components/bench/`
  (per-repo README/ARCHITECTURE now live under `docs/components/<comp>/`);
  the two "bench never imports sim" invariants made pollution-proof
  (subprocess probe / before-snapshot) — the invariant itself is unchanged;
  three "sim not installed" degradation tests skipped (§6.7 — the state is
  unreachable); nine REST-endpoint tests in the two mixed files skipped with
  documented reasons; eight studio serve-composition tests likewise.
- **Import-order normalization**: with all components first-party in one
  tree, isort's per-repo third-party/first-party split collapses; `ruff
  check --fix` (I001) reordered imports in ~530 files — mechanical,
  semantics-preserving, and the full suite was re-run green afterwards.
- **Coverage** (whole platform, per-component CI selections): **98%** line
  coverage (5,528 tests passing), over the 95% gate. REST-orphaned library modules re-covered by
  characterization suites (`tests/bench/test_characterization_*.py`,
  `tests/studio/test_characterization_cli.py`); studio/cli.py's residual
  uncovered lines are exactly the dead serve-composition path.

## 11. Execution order

1. `scripts/consolidate.py` — copy + exclusions + patch table (§6.1–6.4, 6.8–6.9).
2. Merged `pyproject.toml` (deps/extras/scripts/entry-points/maturin/pytest/
   coverage/ruff) + `uv lock` — resolve conflicts, iterate.
3. Editable install (`uv pip install -e . --group dev` + needed extras);
   `maturin` builds `_core`.
4. Per-component test runs via `scripts/test.py`, fixing §6 fallout until
   green with each repo's original selection.
5. Coverage run; characterization tests to ≥95%.
6. CI workflows, lint/format config, docs, examples index, DEVELOPMENT.md.
