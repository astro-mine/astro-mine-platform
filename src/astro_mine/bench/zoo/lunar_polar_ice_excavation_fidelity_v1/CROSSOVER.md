# Scaling — `lunar-polar-ice-excavation-fidelity-v1`

## The result

> ## The speedup is not a number. It is a curve.

A DEM contact solver is **O(N^2)** in particles. A graph-network surrogate is
**O(N.k)**, where `k` is a packing density (~5 neighbours here) and *not* a function
of `N`. So the ratio between them is not a constant to be quoted — it **grows with the
bed**. Quoting one speedup without the bed size it was measured at is close to
meaningless.

| N | DEM (ms/step) | surrogate (ms/step) | speedup | mean k | max k |
|---|---|---|---|---|---|
| **90** (this scenario's own bed) | 21.6 | 2.32 | **9.3x** | 4.9 | 6 |
| **250** | 149.2 | 7.18 | **20.8x** | 5.4 | 6 |
| **500** | 841.5 | 18.26 | **46.1x** | 5.6 | 7 |
| **1000** | 3649.0 | 53.50 | **68.2x** | 5.7 | 7 |
| **2000** | 15185.8 | 169.42 | **89.6x** | 5.7 | 7 |

At this scenario's own bed (**N = 90**) the substitution is worth
**9.3x**. At N = 2000 it is worth
**89.6x**, and still climbing — DEM's cost is growing quadratically
while the tier's grows linearly.

Note that **max k** — the largest neighbourhood any particle has — barely moves across
the sweep. That is the whole reason the tier *can* be O(N.k): a particle's neighbour
count is set by how densely spheres pack, not by how many of them there are.

## What this measures, and what it does not

**It measures cost.** Per-step wall-clock of Sim's DEM granular engine against the
published surrogate tier, on the same bed, with everything except the particle count
held to what this scenario's *pinned content* produces:

- blade geometry from the pinned excavator — bed 0.8 m, tool 0.15 m;
- soil from the pinned world — density 1500 kg/m3, friction 0.8391, gravity 1.62093 m/s2.

`N` is the only free variable. Nothing here is a bed someone made up.

**It does not measure accuracy, and must not be read as though it did.** The tier is
trained at one particle count. Its calibrated error bound holds where it was
*validated*, and this sweep says nothing about whether it holds at other `N`.

> A speedup at a bed size the tier was never validated on is a **cost result, not a
> substitution claim.** There is deliberately no `is_claim` in `crossover.json`. The
> claim half — speedup *at a held error bound* — belongs to
> `measure_surrogate_speedup.py`, which refuses to publish a number when the bound
> does not hold.

## Why this curve exists at all

Until astro-mine-surrogate#24 the served ONNX graph ran its message passing over a
**dense `(N, N)` adjacency**. That was a deliberate trade — ONNX handles a
data-dependent edge count poorly, and a dense tensor keeps the shape static — and the
numerics were right. The cost was not: it evaluated the edge encoder and every message
MLP across *all* N^2 pairs, then masked ~99% of them away (175x wasted work at N=1000).

That made the **served** tier O(N^2) — the same asymptotics as the solver it exists to
replace. The measured speedup sat flat at **~2x** from N=90 to N=1000, and no amount of
retraining would have moved it.

A single number would have recorded that as a fact about surrogates. The curve records
it as a fact about an implementation, which is what it was. That is the argument for
publishing a curve.

- **Tier** — `excavation-gns:0.4.0`, bundle `sha256:39b4ae65f026e9b4897a9ccab8d638f7595f0286bc3b0cef4e24d31740068dc5`
- **Host** — python `3.12.3`, `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`

> **The measured tier is gone, and cannot be rebuilt.** `excavation-gns:0.2.0`–`0.5.0` were pruned
> from the workspace registry on 2026-08-08, keeping only `0.6.0` — tags and blobs both, so
> `sha256:39b4ae65…` now resolves nowhere. `excavation-gns` was never published to
> `ghcr.io/astro-mine` either; that registry holds the nine anchor packages and this was not one of
> them. There is no second copy. The digest above stays exactly as written: it is the identity of
> what these numbers were measured against, and relabelling it would make the record wrong as well
> as unreproducible.
>
> Do **not** try to reconstruct it with `publish_surrogate.py --version 0.4.0`. That flag is a
> label, not a checkout. The tier's error budget is calibrated from whatever the code says today,
> and that budget is precisely what separates `0.4.0` from `0.6.0` (see the next note) — so the
> command would publish *today's* model under the measured tier's name, into a slot whose emptiness
> means the script's own republish guard cannot object to it.
>
> **To re-measure the curve, run against `0.6.0`**, which is what `measure_fidelity_crossover.py`
> now defaults to. The next note is the argument for why that is sound rather than a compromise:
> cost is a property of the served graph's structure, and the two revisions share it.

> **On the tier version.** The *claim* half (`RESULTS.md`) is measured against `0.6.0`; this cost
> curve is pinned at `0.4.0`, the artifact it was actually measured against. The difference between
> those revisions is the declared **error budget** and its **horizon** (astro-mine-surrogate#21/#23/#27) —
> not the served graph. Cost is a property of the graph *structure* (O(N·k), k=16), which is identical
> across the two, so re-running here would reproduce the same curve; the pin stays honest to the
> digest the numbers came from rather than being relabelled to a version they were not measured on.

*Generated by `scripts/measure_fidelity_crossover.py`. Machine-readable form:
`crossover.json`.*
