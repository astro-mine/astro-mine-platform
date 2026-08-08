# JSON Schema sources

Canonical JSON Schema definitions for SADF documents, plugin manifests, and
configuration. The schema is the source of truth, not any one implementation.

The Pydantic models in `astro_mine.core.<comp>.model` / `.enums` are **hand-written**
(so they keep their curated docstrings and exact semantics), but they may not drift
from these schemas: `scripts/check_model_drift.py` regenerates a model from each schema
with `datamodel-code-generator` and fails CI if the structure diverges (class/field-name
sets, enum value sets). See [`CONTRIBUTING.md`](../../CONTRIBUTING.md) (RM-P0-CORE-07).

Canonical schemas ship **inside their package** so the loader can read them at
runtime in both editable-dev and wheel installs (loaded via `importlib.resources`).

## The canonical set

Every schema below is declared in `scripts/build_schema_bundle.py` and ships in the
published bundle. The two lists are kept honest by
`tests/test_schema_bundle.py`: a schema that reaches the tree without reaching *both* the
bundle and this table fails CI, so neither can quietly fall behind — which is precisely
how `plan.schema.json` once shipped a release cycle outside the bundle (#50, #52).

| Schema | Path (under `src/`) | Traceability |
|---|---|---|
| **SADF v0.1** | `astro_mine/core/sadf/schema/sadf.schema.json` | RM-P0-CORE-01 |
| **ObjectiveSpec v0.1** | `astro_mine/core/objective/schema/objective.schema.json` | RM-P0-CORE-04 |
| **Message catalog v0.1** (control plane) | `astro_mine/core/messages/schema/messages.schema.json` | RM-P0-CORE-04 |
| **Plugin manifest v0.1** | `astro_mine/core/registry/schema/manifest.schema.json` | RM-P0-CORE-05 |
| **RunProvenance v0.1** | `astro_mine/core/provenance/schema/run_provenance.schema.json` | CX-REPRO |
| **PolicyPackage v0.1** | `astro_mine/core/policy/schema/policy_package.schema.json` | RM-P1-CORE-05 |
| **MissionSpec v0.1** | `astro_mine/core/mission/schema/mission.schema.json` | RFC-0001 |
| **Units / frames / epochs v0.1** | `astro_mine/core/units/schema/units.schema.json` | RFC-0007 |
| **Plan / ContingentPlan v0.1** | `astro_mine/core/plan/schema/plan.schema.json` | RFC-0006 |

Two non-JSON-Schema sources ride in the bundle alongside them:

- **Cap'n Proto hot path** — `astro_mine/core/messages/schema/observation.capnp`. The
  per-tick observation family is Cap'n Proto, not JSON Schema (conventions.md §3).
- **Units conformance vectors** — `astro_mine/core/units/schema/conformance.json`: the data
  form of the RFC-0007 guard contract every binding must pass (RM-P1-CORE-08). Part of the
  pinnable identity, so a consumer pins the exact vectors it validated against.

## Cross-file references (RFC-0007)

`messages` and `mission` do not re-declare frame/epoch/CRS types — they `$ref` the shared
units vocabulary across files, **by its absolute `$id`**
(`https://schemas.astro-mine.org/core/units/v0.1/units.schema.json`).

That `$id` is nominal: nothing serves it, and resolution must work offline. A validator
therefore needs the units schema registered under that URI. In-tree, Python does this via
`astro_mine.core.units.schema_ref.units_registry()` and Rust via `UNITS_REF_URI`
(`tests/test_validator_rust_parity.py` asserts the two agree). **Consumers of the published
bundle do not need either** — see below.

## Published schema bundle (RM-P0-CORE-08)

These JSON Schemas — together with the Cap'n Proto and `.proto` sources — are assembled
into a versioned, **content-addressed bundle** by `scripts/build_schema_bundle.py` and
published as an OCI artifact to private GHCR
(`ghcr.io/astro-mine/astro-mine-platform/schemas:<version>`) by the `publish-schemas` workflow,
**pullable by digest**. The bundle's `schema_digest` (a sha256 over the schema sources) is
the identity a Bench run pins to reproduce a scenario byte-for-byte (`docs/VERSIONING.md`
§4–5; CX-REPRO).

### Pinning the digest

The same digest is shipped **in the package** as `astro_mine.core.SCHEMA_DIGEST`, so a
consumer can pin the exact Core schema set it validated against without pulling the bundle:

```python
from astro_mine.core import SCHEMA_DIGEST   # "sha256:…" — equals the bundle's schema_digest
```

While `CORE_INTERFACE_VERSIONS` stays frozen at `0.1.0` (`VERSIONING.md` §4), version
negotiation is a no-op and this is the value that actually distinguishes one Core schema set
from another — it is what a Bench `ScenarioSpec` pins (§4.2; CX-REPRO).

It is a **generated, committed constant** (`src/astro_mine/core/_schema_digest.py`), not a
runtime recompute, because the digest covers the `.proto` sources under `schemas/proto/` —
which are at the repo root and *not* in the wheel. An installed Core therefore cannot
reproduce the digest from its own files, and a filesystem walk relative to `__file__` would
yield a plausible-but-wrong value in a wheel while looking correct in this repo. Regenerate
it whenever a schema changes (CI fails if you forget):

```bash
uv run python scripts/build_schema_bundle.py --update-digest
```

The bundle is **self-sufficient**: `bundle.json` carries a `schema_index` mapping each
schema's `$id` to its path inside the bundle. Register every entry under its `$id` and a
stock JSON Schema validator resolves the cross-file units `$ref` with no Core-specific code
— which is what lets a non-Python binding validate against the published contract:

```python
index = json.loads((bundle / "bundle.json").read_text())["schema_index"]
registry = Registry().with_resources(
    [(sid, Resource.from_contents(json.loads((bundle / rel).read_text())))
     for sid, rel in index.items()]
)
```
