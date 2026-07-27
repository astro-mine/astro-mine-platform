#!/usr/bin/env python3
"""Validate illumination / PSR against a published lunar reference (outside CI).

worlds.md §10 and the Phase-0 Worlds exit criteria require illumination/PSR regression against
published lunar references **with explicit error budgets**. CI covers the analytic half (a flat
plane, a wall, a crater rim in ``tests/test_illumination.py``); this script is the **real-data**
half, which cannot run in offline CI: build the actual Shackleton-de Gerlache world from the real
LOLA DEM and real SPICE kernels, compute the PSR mask over an epoch window, and check the
permanently-shadowed **area fraction** against a committed published reference within its stated
tolerance.

The reference and its budget live in the repo, not in this file's arguments::

    validation/shackleton_psr.reference.json     # the published value + budget + citation
    validation/shackleton_psr.result.json        # the committed artifact of an actual run

Run it after fetching and ingesting the real products (see ``scripts/fetch_shackleton_dem.py``,
``scripts/fetch_spice_kernels.py``, ``scripts/build_shackleton_anchor.py``)::

    python scripts/validate_illumination.py --terrain out/shackleton/terrain \\
        --metakernel data/spice/metakernel.tm \\
        --reference validation/shackleton_psr.reference.json \\
        --report-json validation/shackleton_psr.result.json

With ``--reference`` the harness configuration (azimuth bins, horizon radius, aberration
correction, epoch window, PSR semantics) is taken **from the reference document**, so the run is
by construction comparable with the value it is checked against — a PSR fraction is meaningless
without the window that defines "permanent". Individual flags still override it for exploration.

Exits non-zero when the computed fraction falls outside the error budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.spice import epoch_from_utc, kernel_pool
from astro_mine.worlds.illumination import (
    HorizonFrame,
    IlluminationModel,
    PsrEpochSemantics,
    PsrReference,
    validate_psr,
)
from astro_mine.worlds.terrain import TerrainModel

_SECONDS_PER_DAY = 86_400.0


def _window(start_utc: str, duration_days: float) -> EpochWindow:
    start = epoch_from_utc(start_utc)
    end = Epoch(
        tdb_seconds=start.tdb_seconds + duration_days * _SECONDS_PER_DAY, scale=TimeScale.TDB
    )
    return EpochWindow(start=start, end=end)


def _resolve(args: argparse.Namespace, harness: dict[str, Any], key: str) -> Any:
    """A CLI flag wins; else the reference document's harness config (keys with no flag, like
    ``resolution_m``, only ever come from the document)."""
    supplied = getattr(args, key, None)
    if supplied is not None:
        return supplied
    return harness.get(key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", type=Path, required=True, help="ingested terrain product dir")
    parser.add_argument("--metakernel", type=Path, required=True, help="SPICE meta-kernel (.tm)")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="committed published-reference document (validation/*.reference.json)",
    )
    parser.add_argument(
        "--report-json", type=Path, help="write the result artifact here (the committed evidence)"
    )
    parser.add_argument(
        "--horizon-store",
        type=Path,
        help=(
            "adopt a published bundle's illumination/horizon.zarr instead of re-deriving the "
            "skyline (issue #46). The store is validated against the parameters resolved here and "
            "rejected if they disagree, so this is a cache, not a shortcut: it turns a ~90-minute "
            "re-derivation into seconds. Omit it to force a from-scratch recompute."
        ),
    )
    # Every knob below defaults to None: unset means "take it from the reference document", so the
    # run stays comparable with the value it is graded against unless deliberately overridden.
    parser.add_argument("--start", help="window start (UTC); overrides the reference")
    parser.add_argument("--duration-days", type=float, help="window length (days)")
    parser.add_argument("--step-hours", type=float, help="sampling step (hours)")
    parser.add_argument("--n-azimuth", type=int, help="horizon-map azimuth bins")
    parser.add_argument("--max-radius-m", type=float, help="horizon search radius (m)")
    parser.add_argument("--abcorr", help="SPICE aberration correction for the Sun")
    parser.add_argument("--horizon-frame", choices=[f.value for f in HorizonFrame])
    parser.add_argument("--semantics", choices=[s.value for s in PsrEpochSemantics])
    parser.add_argument(
        "--tolerance", type=float, help="override the reference's absolute error budget"
    )
    args = parser.parse_args(argv)

    reference = PsrReference.load(args.reference)
    harness = reference.harness
    if args.tolerance is not None:
        reference = PsrReference(
            region=reference.region,
            source=reference.source,
            psr_area_fraction=reference.psr_area_fraction,
            psr_area_km2=reference.psr_area_km2,
            tolerance_area_fraction=args.tolerance,
            tolerance_area_km2=reference.tolerance_area_km2,
            harness=harness,
            notes=reference.notes,
        )

    terrain = TerrainModel.open(args.terrain)
    resolution_m = float(terrain.manifest["grid"]["resolution_m"])
    declared = _resolve(args, harness, "resolution_m")
    if declared is not None and abs(float(declared) - resolution_m) > 1e-9:
        print(
            f"WARNING: terrain is {resolution_m:g} m/px but the reference is comparable at "
            f"{float(declared):g} m/px — the PSR area fraction is resolution-sensitive.",
            file=sys.stderr,
        )

    with kernel_pool(str(args.metakernel)):
        model = IlluminationModel(
            terrain,
            n_azimuth=int(_resolve(args, harness, "n_azimuth")),
            max_radius_m=float(_resolve(args, harness, "max_radius_m")),
            abcorr=str(_resolve(args, harness, "abcorr")),
            horizon_frame=HorizonFrame(_resolve(args, harness, "horizon_frame")),
            horizon_store=args.horizon_store,
        )
        result = model.psr_mask(
            _window(
                str(_resolve(args, harness, "start")),
                float(_resolve(args, harness, "duration_days")),
            ),
            float(_resolve(args, harness, "step_hours")) * 3600.0,
            semantics=PsrEpochSemantics(_resolve(args, harness, "semantics")),
        )

    validation = validate_psr(result, reference, resolution_m=resolution_m)
    print(
        f"PSR area fraction: {validation.psr_area_fraction:.4f} "
        f"({validation.psr_area_km2:.0f} km^2 over {validation.n_cells} cells @ "
        f"{resolution_m:g} m)\n"
        f"reference:         {validation.reference_area_fraction:.4f}  "
        f"[{validation.source}]\n"
        f"|error|:           {validation.error_area_fraction:.4f}  "
        f"budget: {validation.tolerance_area_fraction:.4f}\n"
        f"epochs={validation.n_epochs} semantics={validation.semantics} "
        f"illumination_hash={validation.illumination_hash}",
        file=sys.stderr,
    )
    if args.report_json:
        artifact = validation.to_artifact()
        artifact["harness"] = harness
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report_json}", file=sys.stderr)
    print("PASS" if validation.passed else "FAIL", file=sys.stderr)
    return 0 if validation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
