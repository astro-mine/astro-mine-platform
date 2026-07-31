# astro-mine-spice

**The shared SPICE foundation for [Astro-Mine](https://github.com/astro-mine).**
SPICE-backed frames, epochs, and body/topocentric geometry — the name→geometry
resolution that [Core](https://github.com/astro-mine/astro-mine-core) defers (it forbids
heavy dependencies). Every consumer — [Worlds](https://github.com/astro-mine/astro-mine-worlds)
illumination/PSR, [Link](https://github.com/astro-mine/astro-mine-link) LOS/contact
windows, and later Sim's orbital engine and Transit — shares **one** SPICE implementation
instead of re-deriving it or depending on Worlds' geospatial stack.

> **Status:** Phase 0 — extracted from `astro_mine.worlds.spice` per
> [RFC-0002](https://github.com/astro-mine/docs/blob/main/rfc/0002-shared-spice-foundation.md).
> See the [architecture](https://github.com/astro-mine/docs/blob/main/architecture/spice.md).

## Scope

- **Kernel pool** — `load_metakernel`, `kernel_pool`, `clear_kernels` (fail-loud:
  a missing kernel raises, never a silent default).
- **Time** — `et`, `epoch_from_utc`, `epoch_range` over Core `Epoch`/`EpochWindow`.
- **Geometry primitives** — `body_position` (`spkpos`), `frame_transform` (`pxform`).
- **Topocentric site geometry** — `Site`, `BodyGeometry`, `body_geometry`,
  `sun_geometry`, `earth_geometry`.

The boundary: this package resolves Core's vocabulary into positions, rotations, and
topocentric scalars — and stops. **Window search** (`gfposc`) stays in Link; **terrain
occlusion** (horizon maps, `ray_intersect`) stays in Worlds, exposed through the Core
`WorldProvider` contract.

## Layout

```
src/astro_mine/spice/        # import path: astro_mine.spice
tests/                       # SPICE geometry + kernel tests (synthetic kernels, offline)
```

## Development

Spice is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup.

```bash
python scripts/test.py spice
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
