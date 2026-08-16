# `embargo/` — sealed held-out benchmark seeds (do **not** ship, do **not** disclose)

This directory holds the **embargoed held-out seed sets** for zoo scenarios — the anti-gaming
reserve (`docs/architecture/bench.md §9`). One subdirectory per scenario id, each containing a sealed
`heldout_seeds.json` of the form:

```json
{ "salt": "<hex>", "seeds": [ <ints> ] }
```

## Why it lives outside `src/`

The wheel is built by **maturin** (`pyproject.toml [build-system]`), and `[tool.maturin]` sets
`python-source = "src"` — so the packaged tree is rooted at `src/`, and a directory at the
repository root is not a candidate for it. The one escape hatch is `[tool.maturin] include`, which
force-includes non-`.py` package data; every entry there is under `src/astro_mine`, and `embargo`
is not among them. Nothing here reaches a `pip install` or a published zoo OCI artifact.

Do not take that on trust — build the wheel and look:

```bash
uv build --wheel --out-dir dist          # needs the Rust toolchain; Guard's core is compiled
python -m zipfile -l dist/astro_mine_platform-*.whl | grep -c embargo    # -> 0
```

The wheel's top-level entries are exactly `astro_mine/` and `astro_mine_platform-*.dist-info/`.

The public `ScenarioSpec` in the zoo carries only a **commitment** —
`seeds.heldout_commit`, a `sha256:` over the sealed `{salt, seeds}` payload — which binds the held-out
seeds into the scenario's content hash **without disclosing them**. Evaluation discloses the seeds
only at scoring time (`bench.md §5, §9`).

## Verify a commitment

```bash
python scripts/bench/seal_heldout_seeds.py <scenario-id>
python scripts/bench/seal_heldout_seeds.py <scenario-id> --verify sha256:<hex>
```

`tests/bench/test_zoo_anchor.py` also asserts the anchor's sealed payload matches its published
`heldout_commit`.

## SECURITY — rotate before the repo flips public

These seeds are committed to the **private** incubation repo for CI verifiability only. Phase-0 has no
encrypted, eval-time-only disclosure yet (a CX-SEC follow-up). **Before the repo is made public,
rotate every held-out set here** and re-publish the affected scenarios as new versions (a new
`heldout_commit` ⇒ a new scenario version; old leaderboards remain valid for their pinned spec).
