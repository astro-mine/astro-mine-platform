# astro-mine-surrogate

**Learned surrogate models for [Astro-Mine](https://github.com/astro-mine).**
GNN emulators for the most expensive physics in
Sim — above all granular media and
excavation contact — served as low-cost, Core-described plugins behind the physics-step
contract. Every surrogate ships a calibrated error bound: **the error is the product**,
not a footnote.

> **Status:** Phase 1. The **foundational contracts** (RM-P1-SURR-01 — `SurrogateModel`,
> `ErrorReport`, and the Core plugin manifest a surrogate publishes itself as), the
> **learned-DEM excavation surrogate** (RM-P1-SURR-02 — a deep-ensemble GNS with
> split-conformal bounds and an enforced trust region), the **datagen → retrain → gated
> promotion** loop (RM-P1-SURR-03), and the **ONNX-served fidelity tier** with drift/OOD
> re-validation triggers (RM-P1-SURR-04) are in. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/surrogate.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Layout

```
src/astro_mine/surrogate/     # import path: astro_mine.surrogate
  enums.py model.py report.py manifest.py wire.py   # the contracts (SURR-01)
  models/    # learned-DEM GNS ensemble, conformal calibration, trust region (SURR-02)
  datagen/   # sampling policy, oracle seam, design, active learning, dataset store (SURR-03)
  eval/      # the coverage + error-budget promotion gate (SURR-03)
  retrain/   # offline retrain + gated promotion (SURR-03)
  serve/     # ONNX export, runtime, publish, fail-closed load, ServedTier (SURR-04)
  drift/     # OOD/drift monitors + re-validation triggers (SURR-04)
tests/                        # mirrors the package layout
```

Two loops, per [surrogate.md §3](https://github.com/astro-mine/docs/blob/main/architecture/surrogate.md).
**Offline (build):** `datagen` labels a design against a high-fidelity oracle → `models`
trains and calibrates → `eval` gates → `retrain` promotes → `serve.export` publishes.
**Inline (use):** Sim loads the served tier and calls `predict` per tick, while `drift`
watches the live queries and schedules re-validation.

## The error contract

Three types are the whole narrow waist a consumer sees. A **`SurrogateModel`** returns a
prediction, a *calibrated per-channel uncertainty*, and a *trust-region flag* on **every**
call. An **`ErrorReport`** is the static, machine-readable bound it ships with — per-channel
RMSE, calibration/coverage curve, and tail behavior, over a declared trust region, measured
against a content-addressed oracle. And the surrogate publishes itself as a **Core
`PluginManifest`** — Surrogate consumes Core's manifest schema rather than inventing one, so
the surrogate-specific facets ride in the manifest's open `attributes` map:

```python
from astro_mine.surrogate import build_surrogate_manifest
from astro_mine.surrogate.wire import error_report_from_wire, error_report_to_wire

report = surrogate.error_report            # the calibrated bound (with its own content hash)
assert error_report_from_wire(error_report_to_wire(report)) == report   # byte-stable Protobuf

manifest = build_surrogate_manifest(       # a Core PluginManifest — no new plugin kind
    name="excavation-gns", version="0.1.0",
    report=report, artifact_digest=bundle.content_hash(),
)
print(manifest.kind, manifest.attributes["error_report_digest"])   # regime_engine, sha256:...
```

Each channel of the report is typed **continuous or categorical**, and the trust region is a
box over any named input domain — so the same three types carry a dynamical-step surrogate
(behind Sim's `env` interface) and a field-query surrogate (behind Worlds' `world_provider`);
the domain picks the Core `PluginKind`. `ErrorReport` is frozen and `extra="forbid"`, exported
as JSON Schema (`schema/`), and canonically serialized to Protobuf so a cross-language
scheduler can read it.

## The learned-DEM excavation surrogate

The MVP domain (`LUNAR-TR-002`): a GNS/MeshGraphNet-style encoder-processor-decoder over a
particle radius-graph, hand-rolled in PyTorch (no torch_geometric), trained as a **deep
ensemble** with pushforward noise injection on the frozen DEM fixture that ships in the wheel.
The ensemble spread is turned into a real bound by **split-conformal** calibration — a
finite-sample marginal coverage guarantee per output channel — and every query is tested
against the **trust region**, the box over the excavation parameters the fixture swept:

```python
import numpy as np
from astro_mine.surrogate.models import build_excavation_surrogate

surrogate = build_excavation_surrogate(seed=0)   # train + calibrate (70/15/15 split)

prediction = surrogate.predict({
    "position": pos,                 # (N, 2)
    "velocity": vel,                 # (N, 2)
    "tool_x": np.array([0.35]),
    "config": np.array([1500.0, 0.55, 0.3, 0.06]),  # density, friction, restitution, tool_speed
})
prediction.fields["position"]              # (N, 2) next-state mean
prediction.field_uncertainty["position"]   # (N, 2) conformal-calibrated half-widths
prediction.in_domain, prediction.ood_margin   # trust flag + signed distance to the boundary
```

Physics-informed inductive biases do the heavy lifting: edges carry **relative** position
(translation invariance), the network predicts an **acceleration** the surrogate integrates
semi-implicitly (frame consistency), and the message/update MLPs are shared across particles
(permutation equivariance). Outside the trust region the surrogate raises its uncertainty and
lowers `in_domain` — it never returns a confident extrapolation. `surrogate.rollout(...)`
autoregressively feeds predictions back in, which is where rollout drift shows up; the
per-horizon RMSE is recorded in the `ErrorReport`.

Training is deterministic for a given seed + fixture, but torch CPU is not bit-portable across
builds, so CI gates by tolerance rather than a bit-exact golden.

## Offline: datagen, retrain, and the promotion gate

`datagen` is the build loop's sampling half: a declarative, content-addressed
**`SamplingPolicy`** (a Sobol / Latin-hypercube / grid space-filling design plus an
active-learning acquisition) labeled against a high-fidelity oracle. The package **never
imports Sim** — the dependency is inverted behind a Core-typed **`RolloutOracle`** seam, the
same way Bench injects Sim's `EpisodeRunner`. CI and every default path use
`reference_rollout_oracle`, a numpy-only deterministic granular proxy; a caller with the
`[surrogate-datagen]` extra supplies the Sim-backed DEM adapter.

```python
from astro_mine.surrogate.datagen import (
    SamplingPolicy, active_learning_round, generate_dataset,
    reference_rollout_oracle, write_dataset,
)
from astro_mine.surrogate.report import Bound

policy = SamplingPolicy(
    parameter_bounds={"density": Bound(low=1400.0, high=1600.0), ...},
    n_initial=16, pool_size=64, n_per_round=4,   # design + active-learning loop
)
dataset = generate_dataset(policy, reference_rollout_oracle, seed=0)
dataset = active_learning_round(policy, reference_rollout_oracle, surrogate, dataset, seed=1)
ref = write_dataset(dataset, "out/datasets", name="dem-excavation", version="0.2.0")
```

Active learning samples **where the surrogate's residual uncertainty is highest**, not
uniformly. Datasets are immutable and content-addressed — Zarr for the particle/field arrays,
Parquet for the tabular config features; `write_dataset` **refuses to overwrite** an existing
`name:version`, and `read_dataset` verifies the content hash fail-closed. The hash is taken
over canonical big-endian array bytes, so it is stable regardless of chunking or encoding.

Retraining is gated. `evaluate_promotion` is the promotion predicate as a first-class object:
every channel's **calibration curve** must hold (an over-confident surrogate — empirical
coverage well below nominal — fails) *and* its **measured error** must sit inside the
consumer's per-channel budget. `retrain_surrogate` trains a new SemVer version, runs the gate,
and only on a pass exports the served bundle with full reproduction provenance:

```python
from astro_mine.surrogate.eval import PromotionCriteria
from astro_mine.surrogate.retrain import retrain_surrogate

result = retrain_surrogate(
    dataset=ref, hyperparameters=TrainConfig(), seed=0,
    prior_version="0.1.0", criteria=PromotionCriteria(),
    code_version="...", env_lockfile_hash="sha256:...", sampling_policy=policy,
)
result.new_version, result.promoted, result.gate.reasons   # "0.2.0", True, ()
```

A gate failure returns `promoted=False` with **no bundle** — retrained weights never silently
enter Sim, though the attempt's provenance is still recorded for audit. The prior version is
never overwritten. On a pass the `Provenance` carries the train/validation dataset hashes, the
seed, the env lockfile, and the hyperparameter and sampling-policy hashes — enough to
reproduce both the model and its `ErrorReport`.

## Inline: the served ONNX tier

A trained surrogate is exported to a **self-contained ONNX graph**: the radius-graph
featurization, the ensemble, the conformal half-width, the trust-region test, and the OOD
inflation all live *inside* the graph, so ONNX Runtime emits the next state **and** its
calibrated uncertainty, `in_domain` flag, and signed margin with no post-processing — the
served tier needs no torch at all. The bundle is content-addressed (a deterministic archive:
sorted entries, zeroed timestamps, no compression), signed, and published as an OCI artifact
to Hub:

```python
from astro_mine.surrogate.serve import (
    ServedTier, export_excavation_surrogate, publish_served_surrogate, resolve_and_load,
)

bundle = export_excavation_surrogate(surrogate)   # OnnxBundle: graph + serve_meta + ErrorReport
published = publish_served_surrogate(bundle, registry, name="excavation-gns",
                                     version="0.1.0", private_key_pem=key)

served = resolve_and_load(registry, published.reference, verifier=verifier)   # fail-closed
```

The load gate is three fail-closed checks, in order: the registry rejects an unsigned,
tampered-digest, or untrusted-key manifest; the bundle's content hash must equal the
`provenance.digest` the signature binds to (so a swapped artifact is caught); and the bundle's
embedded `ErrorReport` must hash to the `error_report_digest` the manifest declares (so the
served bound is exactly the signed one). Any mismatch raises `ServedIntegrityError` and the
surrogate is never constructed. The ORT session is single-threaded and sequential, so the
served path is deterministic — the property Sim's golden gate pins.

**Surrogate produces a tier; Sim decides when to use it.** `ServedTier` is the surrogate-side
reference of that decision over the two error channels — *static* admission against the
declared budget, and *live* per-query fall-back:

```python
tier = ServedTier(served)
tier.admits({"pos_x": 0.01})                    # static: is the declared budget within tolerance?
prediction = tier.advance(query)
tier.should_escalate(prediction, max_uncertainty=0.005)   # live: OOD or over-tolerance?
```

It is deliberately minimal and is **not** Sim's scheduler — Sim re-implements the decision on
its side over the same Core-visible artifacts (the narrow waist: Sim does not import this
package). A channel the surrogate never measured is never admissible.

## Drift monitoring & re-validation

The inline loop's watchdog. A `DriftMonitor` accumulates the signals a `Prediction` already
carries — the `in_domain` flag, the signed `ood_margin`, and a rolling mean of the calibrated
uncertainty — over a sliding window, and applies a **hybrid** `RevalidationPolicy`: a periodic
schedule *plus* drift triggers (out-of-domain rate, a hard margin breach, or uncertainty
drifting above a baseline multiple). When one fires, the monitor publishes a
`RevalidationTrigger` carrying the window statistics that justified it, so the ground-truth
re-validation and active resample are actionable from the event alone:

```python
from astro_mine.surrogate.drift import DriftMonitor, RevalidationPolicy

monitor = DriftMonitor(policy=RevalidationPolicy(max_ood_rate=0.2), baseline_uncertainty=rmse)
if (trigger := monitor.observe(prediction)) is not None:
    print(trigger.reason, trigger.window_ood_rate)   # e.g. DriftReason.OOD_RATE, 0.31
```

The monitor is pure numpy — it watches predictions from the torch surrogate or the served ONNX
tier alike. The sink is an in-process `DriftEventSink` protocol; the NATS/JetStream transport
is the Cloud deployment and is deferred (surrogate.md §6).

## Known limitations

- **`ServedBackend.NATIVE_GRAPH` is declarable but not servable.** The manifest vocabulary
  reserves a native-graph backend for the case where ONNX cannot express an op, and
  `build_surrogate_manifest` will faithfully record it — but **no native-graph serving runtime
  exists**. `OnnxServedSurrogate` is the only served implementation, and `load_served_surrogate`
  / `resolve_and_load` handle only the ONNX bundle layer. The excavation graph exports cleanly,
  so `export_excavation_surrogate` always produces `ServedBackend.ONNX`; if a future op
  regresses the export it raises `OnnxExportError` rather than falling back. Implementing the
  fallback is deferred past Phase 1.
- **Conformal coverage is *marginal*, not conditional.** The calibrated per-step bounds carry a
  finite-sample marginal guarantee; conditional coverage and long-horizon *rollout* coverage
  remain open (surrogate.md §11). Autoregressive rollout error is *reported* per horizon in the
  `ErrorReport` rather than bounded.
- **One shipped domain.** `PhysicsDomain` spans the dynamical-step and field-query families, but
  only `granular_excavation` has a trained model today. A learned illumination field
  (RM-P1-WORLDS-10) is the next consumer of the same contracts.

## Install

The base install is the contract layer plus the learned-DEM model (Core, protobuf, numpy, and
CPU-pinned torch). The heavier surfaces are optional extras, so a consumer that only reads the
`ErrorReport` wire form stays lean:

| Extra | Pulls | Needed for |
|---|---|---|
| `surrogate-serve` | `onnx`, `onnxruntime`, `onnxscript` | `astro_mine.surrogate.serve` — export, ONNX Runtime, load |
| `surrogate-datasets` | `zarr`, `pyarrow`, `scipy` | `astro_mine.surrogate.datagen` / `retrain` — design, store, retrain |
| `surrogate-publish` | (empty marker) | signing + publishing a served surrogate to a registry — Hub is now in-package |
| `surrogate-datagen` | (empty marker; wants `[sim-dem]`) | regenerating the DEM fixture (`scripts/gen_dem_dataset.py`) only |

The `surrogate-datagen` extra is **not** a dependency and is never installed in CI — it exists only so
`scripts/gen_dem_dataset.py` can drive Sim's high-fidelity DEM engine to regenerate the frozen
fixture. The package itself never imports Sim: high-fidelity data arrives through the
`RolloutOracle` seam as a content-addressed dataset.

## Development

Surrogate is part of the [`astro-mine-platform`](../../../README.md) distribution — one
repository, one environment, one test suite. See
[`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup.

```bash
python scripts/test.py surrogate
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
