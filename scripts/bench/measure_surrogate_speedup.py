#!/usr/bin/env python3
"""Measure the DEM-vs-surrogate speedup on a zoo scenario (RM-P1-SURR-04 / RM-P1-SIM-03).

Produces the headline result of `lunar-polar-ice-excavation-fidelity-v1`: what the high-fidelity DEM
granular solver costs, against a learned surrogate standing in for it, **at the error bound the
substitution actually held to** (surrogate.md §8/§12; LUNAR-TR-002). Writes `results.json` +
`RESULTS.md` beside the scenario.

    python scripts/measure_surrogate_speedup.py \
        --registry /path/to/hub-registry \
        --surrogate excavation-gns:0.6.0

## This script is the *only* place Bench touches Sim

Bench is `core` + `pydantic`. It ships no Sim code, declares no `[sim]` extra, and never imports Sim
from library code — the real Sim-backed runner reaches it through the injected `EpisodeRunner` seam
and nothing else (bench.md §2.2; conventions.md §1.1). This file lives **outside `src/`**, is not in
the wheel, and imports Sim lazily behind a clean install hint — the same optional-component pattern
as `scripts/determinism_gate.py`.

And the seam really does hold: `FidelitySpeedupRunner` already satisfies Bench's `EpisodeRunner`
protocol structurally (`__call__(resolved, policy, seed) -> EpisodeTrace`), so it is passed straight
to `bench.baseline.run(...)` with **zero changes to Bench's runner path**. Sim performs the entire
two-tier comparison behind that seam; Bench does not orchestrate two runs and does not know there
were two. The speedup rides out as *runner state* (`runner.reports`), mapped through
`.as_provenance()` into plain dicts — so no Sim type ever crosses into Bench.

## Why the result is a sidecar and not a Scorecard field

Because wall-clock is not deterministic and `Scorecard.content_hash` covers the whole model dump: a
duration in there would break `assert_score_reproducible` on every run. The speedup is published
*beside* the scorecard, in a `PerformanceReport` that is never hashed into it. See
`astro_mine.bench.report._performance` for the full argument.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from astro_mine.bench.baseline import BaselinePolicy, run
from astro_mine.bench.report import PerformanceReport, SurrogateIdentity, performance_report
from astro_mine.bench.zoo import load_scenario

#: The scenario whose headline result this is.
SCENARIO_ID = "lunar-polar-ice-excavation-fidelity-v1"

_INSTALL_HINT = """\
this measurement needs the Sim-backed runner, which Bench deliberately does not depend on.
Install the measurement environment (Bench stays core+pydantic; Sim reaches it through the
injected EpisodeRunner seam):

    uv pip install 'astro-mine-sim[dem,hub]' astro-mine-worlds astro-mine-prospect onnxruntime

`astro-mine-worlds` is not optional here: without it no `world_provider` entry point is
discovered, the pinned world silently resolves to `None`, and Sim falls back to its
reduced-order regolith constants — measuring the surrogate against the wrong soil.
"""


def _load_sim() -> Any:
    """Import the Sim surface this script needs, or exit with an install hint."""
    try:
        from astro_mine.sim.bench import FidelitySpeedupRunner
        from astro_mine.sim.engines.surrogate import load_surrogate_tier
        from astro_mine.sim.runtime._hub_adapter import open_bundle_store
    except ImportError as exc:
        print(f"{exc}\n\n{_INSTALL_HINT}", file=sys.stderr)
        raise SystemExit(2) from exc
    return FidelitySpeedupRunner, load_surrogate_tier, open_bundle_store


def _memoized_world_factories() -> dict[str, Any]:
    """The provider factories, with the **world** provider memoized by content.

    Sim's `ContentResolver` caches providers by digest, but it is constructed fresh per episode, so
    the cache never survives a seed: the two tiers and five seeds here would otherwise rebuild the
    same world ten times.

    That used to be the difference between running this measurement and not. The anchor world
    shipped no horizon map, so every `from_bundle` re-derived a 1264 x 1264 x 120 skyline — a
    192-million-entry ray-march — from the packaged DEM, and ten of them cost the better part of a
    day. World `0.4.0` persists the skyline and the load path adopts it (astro-mine-worlds#46), so a
    build is now ~3 s and memoizing saves 30 s rather than a day. Kept anyway: it is free, and the
    reason it was sound has not changed.

    It cannot flatter the result either way. Provider construction is *setup*, entirely outside the
    `advance` bracket that Sim's `TimedEngine` measures and that the speedup is a ratio of, and the
    provider is a pure function of content-addressed bytes, read-only during a run. It buys
    wall-clock for the operator, not for the surrogate.
    """
    from astro_mine.sim.runtime.content import _discover_factories

    discovered = _discover_factories()
    if "world_provider" not in discovered:
        print(
            "no `world_provider` entry point is installed: the pinned world would resolve to None "
            "and Sim would fall back to its reduced-order regolith constants, measuring the "
            f"surrogate against the wrong soil.\n\n{_INSTALL_HINT}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    build_world = discovered["world_provider"]
    cache: dict[str, Any] = {}

    def memoized_world(manifest: Any, layers: Any) -> Any:
        key = "" if manifest.provenance is None else manifest.provenance.digest
        if key not in cache:
            start = time.perf_counter()
            cache[key] = build_world(manifest, layers)
            source = getattr(cache[key].illumination, "horizon_source", "?")
            print(
                f"  world provider built in {time.perf_counter() - start:.1f}s "
                f"(horizon: {source}) — setup, outside the timed bracket",
                flush=True,
            )
        return cache[key]

    return {**discovered, "world_provider": memoized_world}


def _package_versions() -> dict[str, str]:
    """The versions of the code that produced the physics on both sides of the ratio."""
    from importlib.metadata import PackageNotFoundError, version

    stamped: dict[str, str] = {}
    for name in (
        "astro-mine-bench",
        "astro-mine-sim",
        "astro-mine-core",
        "astro-mine-worlds",
        "astro-mine-hub",
        "numpy",
        "onnxruntime",
    ):
        try:
            stamped[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - an absent optional component
            continue
    return stamped


def _render_markdown(report: PerformanceReport, *, revalidate_every: int) -> str:
    """The human-readable result — the number, and every bound it has to be read against."""
    headline = report.headline_speedup
    claim = f"**{headline:.2f}x**" if headline is not None else "**no claim** (no seed qualified)"
    lines = [
        f"# Measured result — `{report.scenario_id}`",
        "",
        "## The number",
        "",
        f"> ## {claim}",
        ">",
        "> DEM granular solver wall-clock / learned-surrogate wall-clock, median over the seeds "
        "that support a claim.",
        "",
        "A speedup is not a result on its own. It is a result *at an error bound*, produced by a "
        "*specific* artifact, on a *specific* machine. All three are below.",
        "",
        "## What it was measured against",
        "",
        f"- **Surrogate under test** — `{report.surrogate.name}:{report.surrogate.version}`",
        f"  - bundle digest: `{report.surrogate.content_hash}`",
        f"  - calibrated `ErrorReport`: `{report.surrogate.error_report_digest}`",
        f"  - sampling policy (the box that *is* its trust region): "
        f"`{report.surrogate.sampling_policy_hash}`",
        f"- **Task** — `{report.scenario_id}`, spec hash `{report.spec_hash}`",
        f"- **Runner** — `{report.runner}`",
        f"- **Re-validation** — every {revalidate_every} steps the surrogate's step is compared "
        "against a DEM reference bed stepped from the same state. That DEM work is inside the "
        "surrogate tier's own wall-clock, so this ratio is **conservative**: it charges the "
        "surrogate for the cost of proving it was still right.",
        "",
        "## The error bounds",
        "",
        "Three different bounds, kept apart because they answer different questions.",
        "",
        "| channel | realized (worst) | task tolerance (admitted) | declared by the artifact |",
        "|---|---|---|---|",
    ]
    first = report.seeds[0]
    worst = report.worst_realized_error
    channels = sorted(set(worst) | set(first.admitted_tolerance) | set(first.declared_error_budget))

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.6g}"

    for channel in channels:
        lines.append(
            f"| `{channel}` "
            f"| {cell(worst.get(channel))} "
            f"| {cell(first.admitted_tolerance.get(channel))} "
            f"| {cell(first.declared_error_budget.get(channel))} |"
        )
    lines += [
        "",
        "- **realized** — what the surrogate actually deviated by, worst case across claiming "
        "seeds, measured against a DEM reference bed.",
        "- **task tolerance** — what the run was *admitted* under and re-validated against. "
        "`LUNAR-TR-002`'s operative bound: Sim must refuse substitution beyond **task** tolerance.",
        "- **declared** — what the artifact advertises it holds to (its calibrated `ErrorReport`). "
        "A property of the tier, fixed before this measurement existed.",
        "",
        "## Per seed",
        "",
        "| seed | speedup | admitted | within budget | escalated | holds declared | claim? |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed in report.seeds:
        speed = f"{seed.speedup:.2f}x" if seed.speedup is not None else "—"
        lines.append(
            f"| {seed.seed} | {speed} | {seed.admitted} | {seed.within_budget} "
            f"| {seed.escalated} | {seed.holds_declared_budget} | "
            f"{'**yes**' if seed.is_claim else 'no'} |"
        )
    lines += [
        "",
        "A seed is a **claim** only if the surrogate was admitted, held the tolerance it ran "
        "under, "
        "and never escalated back to the reference solver. A ratio from a run that escalated is "
        "partly the reference solver against itself — a number with the shape of a result and none "
        "of the content — so the headline aggregates over claiming seeds only.",
        "",
        "## Wall-clock, per tier",
        "",
        "| seed | DEM `advance` (s) | surrogate `advance` (s) | DEM steps | surrogate steps |",
        "|---|---|---|---|---|",
    ]
    for seed in report.seeds:
        lines.append(
            f"| {seed.seed} | {seed.dem.advance_wall_clock_s:.3f} "
            f"| {seed.surrogate.advance_wall_clock_s:.3f} "
            f"| {seed.dem.steps} | {seed.surrogate.steps} |"
        )
    lines += [
        "",
        "The bracket is `advance` and nothing else — not scenario resolution, not content pulls, "
        "not world construction, not bed settling. The ratio is a comparison of **physics cost**, "
        "which is the only thing a surrogate claims to change.",
        "",
        "## The host",
        "",
        "A speedup is a ratio, so it cancels most host effects — but not all. The two tiers stress "
        "different code (a numpy O(N^2) DEM kernel vs. an ONNX Runtime session) and do not scale "
        "alike with core count or SIMD width. This number is a measurement *of this machine*.",
        "",
        f"- python `{report.toolchain.python}`",
        f"- platform `{report.toolchain.platform}`",
        f"- cpu_count `{report.toolchain.cpu_count}`",
        "",
        "| package | version |",
        "|---|---|",
    ]
    for name, ver in sorted(report.toolchain.packages.items()):
        lines.append(f"| `{name}` | {ver} |")
    lines += [
        "",
        "---",
        "",
        "*Generated by `scripts/measure_surrogate_speedup.py`. The machine-readable form is "
        "`results.json`. This result is **not** part of the Scorecard and is never folded into its "
        "`content_hash` — see `astro_mine.bench.report._performance`.*",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry", required=True, type=Path, help="local OCI-layout Hub registry"
    )
    parser.add_argument(
        "--surrogate",
        default="excavation-gns:0.6.0",
        help="surrogate tier reference (name:version)",
    )
    parser.add_argument(
        "--metakernel",
        required=True,
        type=Path,
        help=(
            "SPICE meta-kernel (.tm). Required: the excavator now stands on the pinned world's "
            "actual surface (astro-mine-sim#58), so resolving its site evaluates the world's "
            "illumination — and that needs ephemerides. Before that fix the asset sat 25 m from "
            "the Moon's centre, every world query fell through to an out-of-grid default, and "
            "SPICE was never reached."
        ),
    )
    parser.add_argument(
        "--pub", type=Path, help="trusted public-key PEM; pins the tier's signature to a known key"
    )
    parser.add_argument("--scenario", default=SCENARIO_ID, help="zoo scenario id")
    parser.add_argument(
        "--revalidate-every",
        type=int,
        default=None,
        help=(
            "steps between DEM re-validations. Defaults to the tier's declared budget horizon "
            "(`budget_horizon_steps`): the budget only bounds drift over that many steps, and the "
            "engine refuses a coarser cadence (surrogate#23). Pass a value only to re-validate "
            "*more* often than the tier requires."
        ),
    )
    parser.add_argument(
        "--out", type=Path, help="directory for results.json + RESULTS.md (default: the zoo entry)"
    )
    args = parser.parse_args(argv)

    FidelitySpeedupRunner, load_surrogate_tier, open_bundle_store = _load_sim()

    from astro_mine.core.registry import PluginManifest

    spec = load_scenario(args.scenario)
    print(f"scenario {spec.scenario_id}  spec_hash {spec.spec_hash}", flush=True)

    # Read the surrogate tier off the registry and load it **fail-closed**.
    #
    # Note the read goes through `Registry`, not through Hub's `BundleStore`/`HubClient` path that
    # Sim resolves world/fleet/prospect content with. That is not a shortcut: a surrogate artifact
    # is verified by the **signature embedded in its own PluginManifest** (which
    # `load_surrogate_tier` checks via Core's `PluginRegistry`, then re-checks the bundle's hash
    # against the signed `provenance.digest`), whereas `HubClient.pull(verify=True)` demands OCI
    # *referrer* attestations — and `publish_served_surrogate` attaches none. Routing a surrogate
    # through the client path therefore fails with "no cosign signature attached" even for a
    # perfectly well-signed tier. Filed as a follow-up; the load below is still fail-closed on the
    # signature, the trusted key, and the content hash, which is what the gate is for.
    from astro_mine.hub.registry import Registry
    from astro_mine.hub.supply_chain import make_verifier

    store = open_bundle_store(args.registry)
    registry = Registry(args.registry)
    descriptor = registry.resolve(args.surrogate)
    manifest = PluginManifest.model_validate_json(registry.read_config(descriptor.digest))
    blobs = registry.read_manifest(descriptor.digest)["layers"]
    if len(blobs) != 1:
        print(
            f"expected exactly one layer on {args.surrogate!r}, found {len(blobs)}",
            file=sys.stderr,
        )
        return 2
    bundle_bytes = registry.pull_blob(blobs[0]["digest"])
    verifier = make_verifier(trusted_public_key_pem=args.pub.read_bytes()) if args.pub else None
    surrogate = load_surrogate_tier(bundle_bytes, manifest, verifier=verifier)
    attributes = manifest.attributes or {}
    print(
        f"surrogate {manifest.name}:{manifest.version}\n"
        f"  bundle    {manifest.provenance.digest if manifest.provenance else '?'}\n"
        f"  trust     {json.dumps(surrogate.trust_region, sort_keys=True)}\n"
        f"  budget    {json.dumps(surrogate.recommended_error_budget, sort_keys=True)}",
        flush=True,
    )

    runner = FidelitySpeedupRunner(
        store=store,
        surrogate=surrogate,
        provider_factories=_memoized_world_factories(),
        horizon_steps=spec.episode.horizon_steps,
        revalidate_every=args.revalidate_every,
    )
    # The cadence the run actually used — resolved by the runner from the tier's declared horizon
    # when `--revalidate-every` was left unset, so the recorded/rendered number is the real one, not
    # a `None` placeholder.
    revalidate_every = runner._revalidate_every

    # Bench's ordinary scoring path, with Sim's two-tier runner slotted into the `runner` seam.
    # `run()` is unmodified: it sees an EpisodeRunner returning one EpisodeTrace per seed, and has
    # no idea two tiers were run. The speedup rides out on the runner, not through the trace.
    #
    # The kernel pool is not incidental. The excavator stands on the pinned world's real
    # surface (astro-mine-sim#58), so resolving its site evaluates that world's illumination, and
    # that needs ephemerides. It is worth knowing why this was not here before: the asset used to
    # be placed 25 m from the Moon's *centre*, so every world query fell through to an out-of-grid
    # default and SPICE was never reached — along with the world's gravity and regolith, which is
    # what made the DEM bed NaN and this measurement unable to produce a number at all.
    from astro_mine.spice import kernel_pool

    print(f"\nrunning seeds {list(spec.seeds.public)} at both tiers...", flush=True)
    with kernel_pool(str(args.metakernel)):
        scorecard = run(spec, BaselinePolicy(), runner=runner)
    print(f"scorecard content_hash {scorecard.content_hash}", flush=True)

    report = performance_report(
        {seed: r.as_provenance() for seed, r in runner.reports.items()},
        scenario_id=spec.scenario_id,
        spec_hash=spec.spec_hash,
        runner=getattr(runner, "__name__", type(runner).__name__),
        surrogate=SurrogateIdentity(
            name=manifest.name,
            version=manifest.version,
            content_hash=manifest.provenance.digest if manifest.provenance else None,
            error_report_digest=attributes.get("error_report_digest"),
            sampling_policy_hash=(
                (manifest.provenance.source_content_hashes or {}).get("sampling_policy")
                if manifest.provenance
                else None
            ),
        ),
        packages=_package_versions(),
    )

    out = args.out or (
        Path(__file__).resolve().parent.parent
        / "src/astro_mine/bench/zoo"
        / args.scenario.replace("-", "_")
    )
    out.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    # The derived headline is the *published* number, so it is written down rather than left for
    # every reader to re-derive (and re-derive differently).
    payload["headline_speedup"] = report.headline_speedup
    payload["worst_realized_error"] = report.worst_realized_error
    payload["is_claim"] = report.is_claim
    payload["revalidate_every"] = revalidate_every
    # The Scorecard's hash is recorded *beside* the timing, never inside it — the same sibling
    # relationship Sim's MCAP envelope gives timing next to `content_hash`.
    payload["scorecard_content_hash"] = scorecard.content_hash
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "RESULTS.md").write_text(
        _render_markdown(report, revalidate_every=revalidate_every), encoding="utf-8"
    )

    headline = report.headline_speedup
    print(
        f"\n{'=' * 68}\n"
        f"  speedup (median over claiming seeds): "
        f"{f'{headline:.2f}x' if headline is not None else 'NO CLAIM'}\n"
        f"  claiming seeds: {len(report.claiming_seeds)}/{len(report.seeds)}\n"
        f"  worst realized error: {json.dumps(report.worst_realized_error, sort_keys=True)}\n"
        f"  declared bound:       "
        f"{json.dumps(dict(report.seeds[0].declared_error_budget), sort_keys=True)}\n"
        f"{'=' * 68}\n"
        f"wrote {out / 'results.json'} and {out / 'RESULTS.md'}"
    )
    return 0 if report.is_claim else 1


if __name__ == "__main__":
    raise SystemExit(main())
