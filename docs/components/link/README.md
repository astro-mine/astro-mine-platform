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

> **Where the commands live.** This package ships no console scripts. Link's commands are
> `astro-mine link publish`, provided by
> [`astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli) — a separate distribution that
> depends on this one. There is one executable and one grammar
> ([`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
> [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md)); the earlier
> `astro-mine-link` binary and its bare alias are both retired.

## Layout

```
src/astro_mine/link/        # import path: astro_mine.link
  geometry/ windows/ budget/ products/ cache/
  constellation/ network/   # relay constellation, multi-hop reachability, DTN
  anchor/                   # the anchor scenario's pinned comms model (relay + DSN)
  registry/                 # Core `comms_model` plugin manifest + Hub publish/resolve
scripts/                    # maintainer tools (build + publish the anchor contact plan)
tests/link/                 # mirrors the package layout
```

## Publishing a contact plan

A `ContactPlan` is a content-addressed [Hub](https://github.com/astro-mine/docs/blob/main/architecture/hub.md)
artifact: Link declares itself a Core `comms_model` plugin, and a consumer (Sim, Bench) resolves the
plan **by content hash** and rebuilds a live `ConnectivitySampler` through the `astro_mine.providers`
entry point — without importing `astro_mine.link`. Everything is offline (a local OCI-layout
registry; no hosted Hub).

```bash
uv sync --extra link-hub
astro-mine hub keygen --out ./keys          # one keygen on the platform; Link has no verb of its own
astro-mine link publish plan.pb --registry ./registry \
    --name astro-mine.link.lunar-polar-relay-dsn --version 0.1.0 \
    --scenario-id lunar-polar-ice-prospecting-v1 --key ./keys/cosign.key
```

The anchor scenario's relay + DSN plan is built end-to-end (real SPICE kernels + the published
Shackleton–de Gerlache terrain bundle, pulled from Hub by digest) by
`scripts/build_anchor_contact_plan.py`; `scripts/build_relay_spk.py` writes the notional relay's SPK
from its pinned orbital elements.

## Development

Link is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup.

```bash
python scripts/test.py link
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
