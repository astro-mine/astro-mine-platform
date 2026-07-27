# Architecture

`astro-mine-surrogate` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/surrogate.md`](https://github.com/astro-mine/docs/blob/main/architecture/surrogate.md):
  purpose, principles, runtime, data, integration, security, and roadmap alignment for
  this package (learned surrogate models with calibrated error bounds — GNN
  granular/excavation emulators behind the Core physics-step contract).
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).

## As built

The Phase-1 backlog (`RM-P1-SURR-01`…`-04`) is complete. How the shipped package realizes
the [`surrogate.md` §3](https://github.com/astro-mine/docs/blob/main/architecture/surrogate.md)
module design — see the [README](README.md) for the usage-level tour:

| `surrogate.md` §3 module | As built | Notes |
|---|---|---|
| *(contracts)* | `enums`, `model`, `report`, `manifest`, `wire`, `_schema` | `SurrogateModel` / `ErrorReport` + the Core-`PluginManifest` builder, JSON Schema, Protobuf wire form (`RM-P1-SURR-01`) |
| `datagen/` | `datagen/` | `SamplingPolicy`, Sobol/LHS/grid design, the `RolloutOracle` seam, active learning, the content-addressed dataset store (`RM-P1-SURR-03`) |
| `models/` | `models/` | The learned-DEM GNS deep ensemble (`RM-P1-SURR-02`) |
| `train/` | `models/train.py` | Folded into `models/` rather than a sibling package — one model family, one training loop |
| `uncertainty/` | `models/conformal.py`, `models/trust_region.py` | Split-conformal calibration + the enforced trust region, alongside the model they calibrate |
| `eval/` | `eval/` | The coverage + error-budget promotion gate (`RM-P1-SURR-03`) |
| `drift/` | `drift/` | OOD/drift monitors + hybrid re-validation triggers (`RM-P1-SURR-04`) |
| `serve/` | `serve/` | ONNX export, ORT runtime, Hub publish, fail-closed load, `ServedTier` (`RM-P1-SURR-04`) |
| `registry/` | `manifest.py` + `serve/publish.py` | No `registry` package and **no `SurrogateManifest` subclass**: Surrogate *consumes* Core's `PluginManifest` (which is `extra="forbid"`) and carries its facets in the manifest's open `attributes` map, so no Core change was needed |
| *(none)* | `retrain/` | The offline-retrain + gated-promotion harness (`RM-P1-SURR-03`) |
| `cli/` | — | Not implemented; the package is used as a library |

Two limitations are worth stating here because they are structural, not incidental:

- **`ServedBackend.NATIVE_GRAPH` is declarable but not servable.** The enum member and the
  manifest's `native_graph_fallback` flag exist so a loader knows what it is admitting, but
  no native-graph serving runtime is implemented — ONNX is the only served backend
  (`OnnxServedSurrogate`). The excavation graph exports cleanly; a future op that cannot be
  expressed in ONNX raises `OnnxExportError` rather than falling back.
- **Conformal bounds are marginal, not conditional.** Long-horizon *rollout* coverage
  remains the open question flagged in `surrogate.md` §11; rollout error is reported
  per-horizon in the `ErrorReport`, not bounded.
