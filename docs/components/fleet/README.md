# astro-mine-fleet

**SADF asset library and authoring toolchain for [Astro-Mine](https://github.com/astro-mine).**
A library of parameterizable assets — orbiters, landers, rovers, hoppers, excavators,
haulers, ISRU plants — each authored in the Swarm Asset Description Format (SADF), plus
the CLI, importers, and exporters to create, lint, convert, and package them. Consume the waist,
never widen it.

> **Status:** Phase 0 — the `fleet` authoring toolchain (RM-P0-FLEET-01), the bidirectional
> URDF/SDF/USD importers and exporters with LOD geometry (RM-P0-FLEET-02), the reference library,
> and signed, content-addressed OCI packaging (RM-P0-FLEET-06) are live. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/fleet.md)
> and [Phase-0 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-0-commons-seed.md).

> **Command renamed.** This CLI is `astro-mine-fleet`; the old name `fleet` still works for one
> deprecation cycle, printing a one-line notice to stderr, and is removed at the first
> public-benchmark milestone. The prefix is normative ([`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
> [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md) §5) — it ends the
> `PATH` land-grab of generic names and makes the package↔command mapping guessable.

## The shipped asset roster

Six reference assets ship as package data under `src/astro_mine/fleet/library/`. They are not
illustrations: packaged with `astro-mine-fleet package`, these six **are** six of the nine content
pins in the anchor benchmark scenario (`lunar-polar-ice-prospecting-v1`).

| Asset | Path | Capability tags |
|---|---|---|
| Prospecting rover | `library/surface/prospecting-rover.sadf.yaml` | `mobility.wheeled`, `prospecting.neutron`, `prospecting.nir`, `prospecting.gpr`, `prospecting.drill_assay`, `excavation.drill`, `sensing.imaging`, `sensing.imu`, `sensing.odometry`, `comms.relay`, `power.generation`, `power.storage` |
| Excavator | `library/manipulation/excavator.sadf.yaml` | `mobility.wheeled`, `excavation.bucket`, `sensing.imu`, `sensing.odometry`, `power.generation`, `power.storage` |
| Hauler | `library/logistics/hauler.sadf.yaml` | `mobility.wheeled`, `sensing.imu`, `sensing.odometry`, `power.generation`, `power.storage` |
| ISRU plant | `library/isru/isru-plant.sadf.yaml` | `isru.thermal_extraction`, `isru.purification`, `isru.storage` |
| Lander | `library/orbital/lander.sadf.yaml` | `carrier.dispenser`, `comms.direct_to_earth`, `power.storage` |
| Relay orbiter | `library/orbital/relay-orbiter.sadf.yaml` | `mobility.orbiter`, `comms.relay`, `comms.direct_to_earth`, `power.generation`, `power.storage` |

Load one — it is package data, so this works from an installed wheel, not only a clone:

```python
from importlib.resources import files

text = files("astro_mine.fleet").joinpath("library/surface/prospecting-rover.sadf.yaml").read_text()
print(text[:200])
```

Or validate and package one straight from the CLI:

```bash
python -c "from importlib.resources import files; print(files('astro_mine.fleet').joinpath('library/manipulation/excavator.sadf.yaml'))"
astro-mine-fleet validate <that path>
astro-mine-fleet package  <that path>
```

**Copy one of these rather than starting from `new`** when your vehicle resembles it: they carry
real capability tags, power models, and fidelity profiles, which the scaffold deliberately leaves
empty.

**Why two of them are at 0.2.0.** The anchor pins `excavator` and `isru-plant` at 0.2.0 and the rest
at 0.1.0, and the reason is recorded in the pins themselves:

- `excavator` 0.2.0 declares a `tool` contact element (#38) — without it no library asset reaches
  the granular contact ladder.
- `isru-plant` 0.2.0 declares a `water_gauge` (`resource_storage`, species water, `si_unit` kg)
  (#40). Without it the plant filled a tank nothing could read: Bench scores `water_mass` by
  matching a reading's species and unit, so a full plant was indistinguishable from a swarm that
  had produced nothing.

That is what a version bump means here — a capability the benchmark can observe, not a cosmetic
edit. Older digests stay immutable and still resolve.

## Layout

```
src/astro_mine/fleet/       # import path: astro_mine.fleet
  cli/ importers/ exporters/ geometry/ lint/ library/ fidelity/ packaging/
tests/                      # mirrors the package layout
```

The `fleet` console command is wired to `astro_mine.fleet.cli:main`.

## CLI

Fleet authors content against Core's SADF — it never defines a parallel schema.

```bash
astro-mine-fleet new rover rover.sadf.yaml   # scaffold a minimal, valid SADF asset
astro-mine-fleet validate rover.sadf.yaml    # structural + semantic validation (Core's gate)
astro-mine-fleet lint *.sadf.yaml            # validity rule over many docs (--json for diagnostics)
astro-mine-fleet resolve rover.sadf.yaml     # emit the canonical JSON form
astro-mine-fleet package rover.sadf.yaml     # write a content-addressed (sha256) asset bundle
```

Signed, content-addressed OCI packaging (RM-P0-FLEET-06) — the pre-Hub distribution
that upgrades to Hub publish in P1:

```bash
astro-mine-fleet keygen --out keys/                              # ECDSA P-256 asset-signing keypair
astro-mine-fleet package rover.sadf.yaml --out dist/oci --oci \
      --sign --key keys/asset-signing.key             # signed OCI artifact, addressed by digest
astro-mine-fleet verify dist/oci --pub keys/asset-signing.pub    # re-hash bytes + verify signature, then load
```

The OCI artifact wraps the SADF wire form + geometry as layers with the Core plugin
manifest as its config (`application/vnd.astro-mine.asset.v1`); the signature (Core's
cosign-modeled envelope, **not** a cosign artifact) attaches as an OCI referrer. `fleet
verify` re-hashes every packaged blob against its content address and the signed digest,
then loads the manifest through Core's `PluginRegistry`.

Phase 0 delivers signed OCI + content provenance (geometry/source hashes). Real
cosign-keyless signatures, SLSA build provenance, and SBOM attestations land with Hub in
Phase 1 — they hang on the same OCI referrer hook, so the upgrade is additive.

Once assets are published to a Hub registry (`astro-mine-fleet publish … --registry <path>`),
`astro-mine-fleet catalog` surfaces that registry as the **selectable robot menu** — the same catalog
Studio renders and Mind/Allocate read capability declarations from (RM-P1-FLEET-11):

```bash
astro-mine-fleet catalog --registry hub/                       # the menu: each asset's kind + capability tags
astro-mine-fleet catalog --registry hub/ --requires mobility.wheeled,excavation.drill
astro-mine-fleet catalog --registry hub/ --preview rover:0.1.0  # a selected asset's glTF preview geometry refs
astro-mine-fleet catalog --registry hub/ --preview rover:0.1.0 --materialize served/  # SADF JSON + glTF → documentUrl
```

A Hub-published vehicle type appears here with **no Fleet code change** — the contract is the
Core capability vocabulary + the Hub catalog, so new kinds arrive as content, not code.

### Import / export / render (RM-P0-FLEET-02)

Fleet converts **both ways** between SADF and the robot-description formats the wider ecosystem
speaks — the bidirectional converters `fleet.md` §11 asks for, with SADF authoritative:

```bash
astro-mine-fleet import rover.urdf -o rover.sadf.json      # URDF / SDF / USD  ->  SADF + USD/glTF geometry
astro-mine-fleet export rover.sadf.json -o rover.urdf                    # SADF -> URDF (+ body-frame OBJ meshes)
astro-mine-fleet export rover.sadf.json -o rover.sdf --format sdf        # SADF -> SDF  (Gazebo)
astro-mine-fleet export rover.sadf.json -o rover.usda --format usd       # SADF -> USD stage (Sim/Studio/Isaac)
astro-mine-fleet render rover.sadf.json -o preview.glb                   # a composed, posed preview/thumbnail
```

**Every export is lossy, and says so.** No robot-description format can hold a spacecraft's power
budget, thermal envelope, sensor observation models, or capability tags — which is exactly why SADF
exists. `astro-mine-fleet export` therefore *always* reports what the target could not carry, as structured
diagnostics (`--json` gives `rule` / `path` / `message`, the same shape `astro-mine-fleet lint` emits):

```bash
astro-mine-fleet export rover.sadf.json -o rover.urdf --json | jq '.losses[].rule'
# "asset.block_dropped"  "joint.effort_unit"  "urdf.lod_dropped"  ...
```

The full **fidelity contract** — what each of the six converter directions preserves and what it
cannot — is `astro_mine.fleet.exporters.LOSS_CONTRACT`, and a test asserts no exporter may report a
lossy edge that is not written down there. Round-trip tests (`tests/test_roundtrip.py`) hold
URDF→SADF→URDF, SDF→SADF→SDF, and USD→SADF→USD to the kinematic tree, masses, and frames, and prove
each loop is a **fixed point** rather than something that erodes an asset a little on every lap.

`--fidelity` dials the visual **LOD tier**: geometry processing emits a full-resolution mesh plus
two decimated tiers per link (alongside the convex collision hull), and a fidelity profile picks one
— `massmodel` takes the coarsest, `articulated` the finest. URDF and SDF carry a single tier and say
which; USD carries the whole ladder.

`astro-mine-fleet render` composes an asset's visual geometry, posed by its frame tree, into one self-contained
glTF or USD file — the preview Studio's robot menu and View's `<AssetPreview/>` widget show. It
needs **no GPU, no renderer, and no network**, so the local tier always works. An asset that
declares mass but no meshes (every Phase-0 reference asset) is previewed with its
**inertia-equivalent proxy boxes** — same mass, same inertia tensor — and the substitution is
reported, never passed off as geometry the asset claims.

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**.

```bash
conda create -n astro-mine-fleet python=3.12
conda activate astro-mine-fleet
uv sync && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
