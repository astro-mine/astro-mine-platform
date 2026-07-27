# Developing astro-mine-platform

## Prerequisites

- **Python 3.12** (pinned in `.python-version`; the machine default 3.13 drifts
  `uv venv` and cannot load the cp312 wheel).
- **uv** for dependency management (the lockfile is `uv.lock`).
- **Rust toolchain** (`rustup`; `source ~/.cargo/env`) — the wheel bundles
  Guard's PyO3 safety core (`astro_mine.guard._core`), built by maturin.

## Setup

```bash
uv sync --group dev        # venv + editable install + dev tooling
```

`uv sync` invokes maturin, which compiles `rust/` and installs the extension —
the first build takes a few minutes.

## Running tests

Each source repo had its own default pytest selection; a single rootdir has a
single `addopts`, so use the runner that re-applies each component's original
default:

```bash
python scripts/test.py                # everything, per-component defaults
python scripts/test.py core sim       # a subset
python scripts/test.py sim -- -k dem  # extra pytest args after --
```

Or run pytest directly with a component's CI selection (see
`.github/workflows/ci.yml` for the exact `-m` expression per component):

```bash
uv run python -m pytest tests/guard -m "not sim and not slow"
```

Opt-in lanes (need services/hardware, self-skip or are marker-gated):
`-m sim` (real-Sim e2e for guard/mind), `-m cluster` (live k8s / Ray cluster),
`-m integration` for hub (live Postgres/OCI), `-m scale` (allocate benchmark),
`-m postgres|minio|nats|mlflow|docker|container|gpu|realdata`.

Coverage across the whole platform:

```bash
bash scripts/coverage_sweep.sh
```

## Lint / format

```bash
uv run ruff check src tests scripts
uv run ruff format src tests scripts
```

(CI lints with the latest ruff via `uv run ruff` — do not pin `uvx ruff`.)

## Re-running the consolidation

`scripts/consolidate.py` re-copies all 18 component repos from the sibling
clones under `../` and re-applies the documented transforms
(docs/CONSOLIDATION_PLAN.md §6). Manual edits listed in the plan but not
encoded in the script (REST seam errors, skip marks, test rescopes) live in
this repo's history — re-running the script over them will clobber those files;
diff against git before committing a re-run.

## Layout

See README.md and docs/CONSOLIDATION_PLAN.md §2. The unusual bits:

- Root dirs like `embargo/`, `validation/`, `platform/`, `schemas/proto` are
  **load-bearing paths** — src code and committed digests resolve them
  root-relatively. Do not move them.
- `dist/schema-bundle/` is a *committed* artifact (core's schema digest), not a
  build output; `.gitignore` carves it out.
- `tests/` is a package: `tests/<comp>/` mirrors each source repo's suite.
