# astro-mine-link

**Communications environment for [Astro-Mine](https://github.com/astro-mine).**
Line-of-sight and terrain occlusion, relay/ground-station contact windows, parametric
link budgets, and the observation masks that make coordination hard — the constraint
that makes the anchor scenario comms-denied *for real*. Geometry is ground truth; RF
is a layer on top.

> **Status:** Phase 0 — scaffolding (MVP scope). Seeded and ready for the Phase-0
> backlog. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/link.md)
> and [Phase-0 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-0-commons-seed.md).

> **Command renamed.** This CLI is `astro-mine-link`; the old name `link` still works for one
> deprecation cycle, printing a one-line notice to stderr, and is removed at the first
> public-benchmark milestone. The prefix is normative ([`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
> [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md) §5) — it ends the
> `PATH` land-grab of generic names and makes the package↔command mapping guessable.

## Layout

```
src/astro_mine/link/        # import path: astro_mine.link
  geometry/ windows/ budget/ products/ cache/
  constellation/ network/   # relay constellation, multi-hop reachability, DTN
  anchor/                   # the anchor scenario's pinned comms model (relay + DSN)
  registry/                 # Core `comms_model` plugin manifest + Hub publish/resolve
  cli.py                    # `astro-mine-link publish` / `astro-mine-link keygen`
scripts/                    # maintainer tools (build + publish the anchor contact plan)
tests/                      # mirrors the package layout
```

## Publishing a contact plan

A `ContactPlan` is a content-addressed [Hub](https://github.com/astro-mine/docs/blob/main/architecture/hub.md)
artifact: Link declares itself a Core `comms_model` plugin, and a consumer (Sim, Bench) resolves the
plan **by content hash** and rebuilds a live `ConnectivitySampler` through the `astro_mine.providers`
entry point — without importing `astro_mine.link`. Everything is offline (a local OCI-layout
registry; no hosted Hub).

```bash
uv sync --extra hub
uv run astro-mine-link keygen --out ./keys
uv run astro-mine-link publish plan.pb --registry ./registry \
    --name astro-mine.link.lunar-polar-relay-dsn --version 0.1.0 \
    --scenario-id lunar-polar-ice-prospecting-v1 --key ./keys/cosign.key
```

The anchor scenario's relay + DSN plan is built end-to-end (real SPICE kernels + the published
Shackleton–de Gerlache terrain bundle, pulled from Hub by digest) by
`scripts/build_anchor_contact_plan.py`; `scripts/build_relay_spk.py` writes the notional relay's SPK
from its pinned orbital elements.

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**.

```bash
conda create -n astro-mine-link python=3.12
conda activate astro-mine-link
uv sync && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
