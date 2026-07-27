# Astro-Mine-Platform

The consolidated, single-package distribution of the Astro-Mine Python platform:
every component — Core, Spice, Seal, Worlds, Prospect, Link, Fleet, Sim, Bench,
Cloud, Surrogate, Mind, Learn, Allocate, Guard, Hub (client), Studio (library),
and the `astro-mine` umbrella CLI — in one repository, one distribution, one
`astro_mine` package.

This repository is a **faithful consolidation, not a rewrite**. Code was copied
mechanically from the 18 component repos; import paths (`astro_mine.<comp>`),
public APIs, schemas and their `$id`s, CLI binaries, entry-point groups,
configuration and environment-variable semantics, and algorithms are unchanged.
See [docs/CONSOLIDATION_PLAN.md](docs/CONSOLIDATION_PLAN.md) for the migration
contract, the small list of deliberate deviations (REST API surfaces and the
TypeScript UIs are not migrated), and every place code had to be edited.

## Install

```bash
# dev setup (WSL/Linux; needs Python 3.12, uv, and a Rust toolchain for the
# Guard safety core)
uv venv && uv pip install -e . --group dev
```

Heavy optional stacks stay behind component-prefixed extras, preserving the
"local tier stays light" property (CX-LOCAL) the per-repo extras used to encode:

```bash
uv pip install -e ".[learn-rllib]"     # Ray RLlib + Torch training path
uv pip install -e ".[sim-mujoco]"      # MuJoCo wheel-soil contact engine
uv pip install -e ".[mind-onnx]"       # ONNX learned-controller tier
```

## CLI

All component binaries ship from this one distribution, unchanged:
`astro-mine` (the umbrella), `astro-mine-core`, `astro-mine-sim`,
`astro-mine-bench`, `astro-mine-fleet`, `astro-mine-worlds`, `astro-mine-link`,
`astro-mine-prospect`, `astro-mine-guard`, `astro-mine-mind`,
`astro-mine-learn`, `astro-mine-hub`, `astro-mine-cloud`, `astro-mine-studio`,
plus the one-cycle deprecated aliases (`fleet`, `worlds`, `link`, `prospect`,
`astro-mine-train`).

## Tests

```bash
python scripts/test.py            # every component suite, each with its
                                  # source repo's original default selection
python scripts/test.py core sim   # a subset
```

## Layout

```
src/astro_mine/<comp>/   the 18 component packages, copied verbatim
tests/<comp>/            each repo's test suite
schemas/<comp>/          proto / JSON-schema codegen sources
examples/<comp>/         each repo's runnable examples
rust/                    Guard's Rust safety-core crate (PyO3 → astro_mine.guard._core)
scripts/consolidate.py   the mechanical migration script (re-runnable)
docs/                    consolidation plan + per-component notes
```

## License

Apache-2.0 — © Astro-Mine project contributors.
