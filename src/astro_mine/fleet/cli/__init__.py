"""SADF authoring/validation/lint toolchain + the ``fleet`` CLI (RM-P0-FLEET-01).

The authoring front door, dispatched by stdlib ``argparse`` (no CLI-framework runtime
dependency — Phase-0 minimalism):

- ``new``      — scaffold a minimal, valid SADF asset to a file;
- ``validate`` — structural + semantic validation of one document (Core's gate);
- ``lint``     — Core's gate **plus** physical-plausibility rules over many documents,
                 aggregating diagnostics;
- ``resolve``  — emit a document's canonical JSON form;
- ``import``   — convert a URDF/SDF/USD description into a SADF asset + USD/glTF geometry;
- ``export``   — convert a SADF asset back out to URDF/SDF (ROS) or a USD stage (Sim/Studio),
                 reporting exactly what the target format could not carry;
- ``render``   — compose an asset's geometry into one posed preview/thumbnail (glTF or USD);
- ``fidelity`` — list an asset's multi-fidelity profiles (coarse -> fine);
- ``package``  — write a content-addressed asset bundle, or a signed OCI artifact
                 (``--oci``/``--sign``; RM-P0-FLEET-06);
- ``verify``   — verify an OCI asset artifact's signature and that its manifest loads;
- ``publish``  — publish a signed SADF asset to a [Hub](hub.md) registry (RM-P1-FLEET-10);
- ``catalog``  — list a [Hub](hub.md) registry as the selectable robot menu, or preview one
                 asset's glTF/USD geometry (RM-P1-FLEET-11);
- ``families`` — list the parametric asset families and their parameter ranges;
- ``resolve-family`` — resolve a parametric family + parameter overrides to concrete SADF.

``validate`` is the schema gate; ``lint`` runs that same gate and then the
physical-plausibility rules (positive-definite inertia, power balance, sensor sanity;
RM-P0-FLEET-03) over each document, so an asset that is schema-valid but physically
impossible still fails ``lint``. ``fidelity`` lists the representation tiers an asset
declares under one identity for Sim's scheduler (RM-P0-FLEET-05).

``import`` parses URDF (``yourdfpy``), SDF (stdlib XML), or a USD stage (``pxr``) and writes a
validated SADF document plus normalized USD + glTF geometry with a generated collision hull and
visual LOD tiers (RM-P0-FLEET-02). ``export`` is its mirror — the bidirectional converters
fleet.md §11 asks for — and always prints what the target format could **not** carry (a power
budget, a sensor model, a capability tag): every export is lossy, and none of it is lossy in
silence. ``render`` composes the asset into a single posed preview file, the thumbnail Studio and
View show; it needs no GPU and no network, so the local tier always works (fleet.md §7).

Fleet **consumes the waist**: every SADF type, validator, and the wire form come from
:mod:`astro_mine.core.sadf` — this CLI defines no parallel schema (CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from astro_mine.core.registry import PluginManifest, PluginRegistry, RegistryError
from astro_mine.core.sadf import SadfDocument, SadfError, load_sadf, validate_sadf
from astro_mine.core.sadf.enums import FidelityTier
from astro_mine.fleet import __version__, exporters, fidelity, importers
from astro_mine.fleet._core import CORE_INTERFACES, canonical_json
from astro_mine.fleet.exporters import ExportResult
from astro_mine.fleet.geometry import GeometryError
from astro_mine.fleet.lint import lint_asset
from astro_mine.fleet.packaging import oci, package_asset, package_oci
from astro_mine.fleet.packaging.verifier import make_verifier
from astro_mine.seal import SignatureError

__all__ = ["main"]

_DEFAULT_PACKAGE_OUT = "dist/assets"


class Diagnostic(NamedTuple):
    """A single authoring problem: which file, and a human-readable message."""

    source: str
    message: str


# --- the lint rules --------------------------------------------------------------


def _validate_file(path: str) -> list[Diagnostic]:
    """Validate one SADF file against Core's schema gate; [] means it passed.

    This is the schema/semantic tier (what ``validate`` runs). ``lint`` runs this and
    then the physical-plausibility rules in :func:`_lint_file`.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return [Diagnostic(path, f"cannot read file: {exc.strerror or exc}")]
    try:
        validate_sadf(text)
    except SadfError as exc:
        return [Diagnostic(path, str(exc))]
    return []


def _report(args: argparse.Namespace, diagnostics: list[Diagnostic], ok_message: str) -> int:
    """Render diagnostics (text or ``--json``) and return the process exit code."""
    ok = not diagnostics
    if args.json:
        payload = {
            "ok": ok,
            "diagnostics": [{"source": d.source, "message": d.message} for d in diagnostics],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif ok:
        print(ok_message)
    else:
        for d in diagnostics:
            print(f"{d.source}: {d.message}", file=sys.stderr)
    return 0 if ok else 1


# --- subcommands -----------------------------------------------------------------


def _scaffold(asset_id: str, name: str, version: str, kind: str) -> str:
    """A minimal, valid SADF v0.1 document (scalars JSON-quoted so YAML stays valid)."""
    sadf_ver = CORE_INTERFACES["sadf"]
    return f"""\
# SADF v0.1 asset scaffold — authored against astro-mine-core (sadf {sadf_ver}).
# Validate: `astro-mine-fleet validate <path>` · package: `astro-mine-fleet package <path>`.
sadf_version: "0.1"
asset:
  identity:
    id: {json.dumps(asset_id)}
    name: {json.dumps(name)}
    version: {json.dumps(version)}
    kind: {json.dumps(kind)}
  core_interface_versions:
    sadf: {json.dumps(sadf_ver)}
  # root_frame must name a declared frame once `frames:` is non-empty.
  root_frame: "base"
  # frames:
  #   - {{name: base}}
  # capabilities: []          # Core-owned negotiation vocabulary (CapabilityTag)
  # bodies: []                # mass/inertia; add power:, thermal:, sensors:, comms: as needed
  # fidelity_profiles: []     # massmodel / kinematic / articulated (RM-P0-FLEET-05)
"""


def _cmd_new(args: argparse.Namespace) -> int:
    asset_id = args.id or f"example.{args.kind}"
    name = args.name or f"Example {args.kind.replace('_', ' ').title()}"
    text = _scaffold(asset_id, name, args.asset_version, args.kind)
    # The scaffold must always be valid; fail loud if a future edit breaks that.
    try:
        validate_sadf(text)
    except SadfError as exc:  # pragma: no cover - defensive guard on a constant template
        print(f"internal error: scaffold failed validation: {exc}", file=sys.stderr)
        return 1
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out}: file exists (use --force to overwrite)", file=sys.stderr)
        return 1
    if out.parent != Path():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    return _report(args, _validate_file(args.path), ok_message=f"OK: {args.path} is valid SADF")


def _cmd_lint(args: argparse.Namespace) -> int:
    diagnostics: list[Diagnostic] = []
    for path in args.paths:
        diagnostics.extend(_lint_file(path))
    return _report(args, diagnostics, ok_message=f"OK: {len(args.paths)} file(s) passed lint")


def _load(path: str) -> SadfDocument | Diagnostic:
    """Read + load a SADF document, or a Diagnostic describing why it could not."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return Diagnostic(path, f"cannot read file: {exc.strerror or exc}")
    try:
        return load_sadf(text)
    except SadfError as exc:
        return Diagnostic(path, str(exc))


def _lint_file(path: str) -> list[Diagnostic]:
    """Lint one SADF file: Core's schema gate, then physical-plausibility rules.

    Loads the document once (``_load`` applies the structural + semantic gate); a
    read/schema failure short-circuits before plausibility. Each
    :class:`~astro_mine.fleet.lint.PlausibilityFinding` is folded into a ``Diagnostic``
    as ``"<path>: <message> [<rule>]"`` so the rule id travels with the message.
    """
    loaded = _load(path)
    if isinstance(loaded, Diagnostic):
        return [loaded]
    return [Diagnostic(path, f"{f.path}: {f.message} [{f.rule}]") for f in lint_asset(loaded.asset)]


def _cmd_resolve(args: argparse.Namespace) -> int:
    loaded = _load(args.path)
    if isinstance(loaded, Diagnostic):
        print(f"{loaded.source}: {loaded.message}", file=sys.stderr)
        return 1
    rendered = canonical_json(loaded)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered)
    return 0


def _cmd_package(args: argparse.Namespace) -> int:
    if args.sign and not args.oci:
        print("astro-mine-fleet package: --sign requires --oci", file=sys.stderr)
        return 1
    loaded = _load(args.path)
    if isinstance(loaded, Diagnostic):
        print(f"{loaded.source}: {loaded.message}", file=sys.stderr)
        return 1
    if args.oci:
        return _package_oci(args, loaded)
    bundle = package_asset(loaded, args.out)
    if args.json:
        print(json.dumps({"digest": bundle.digest, "path": str(bundle.path)}, sort_keys=True))
    else:
        print(f"packaged {loaded.asset.identity.id} -> {bundle.digest}")
        print(f"  bundle: {bundle.path}")
    return 0


def _package_oci(args: argparse.Namespace, doc: SadfDocument) -> int:
    sign_key: bytes | None = None
    if args.sign:
        if not args.key:
            print(
                "astro-mine-fleet package: --sign requires --key <private-key.pem>", file=sys.stderr
            )
            return 1
        try:
            sign_key = Path(args.key).read_bytes()
        except OSError as exc:
            print(
                f"astro-mine-fleet package: cannot read key {args.key}: {exc.strerror or exc}",
                file=sys.stderr,
            )
            return 1
    artifact = package_oci(doc, args.out, base_dir=Path(args.path).parent, sign_key=sign_key)
    if args.json:
        print(
            json.dumps(
                {
                    "digest": artifact.digest,
                    "asset_digest": artifact.asset_digest,
                    "path": str(artifact.path),
                    "signed": artifact.signed,
                },
                sort_keys=True,
            )
        )
    else:
        suffix = " (signed)" if artifact.signed else ""
        print(f"packaged {doc.asset.identity.id} -> {artifact.digest}{suffix}")
        print(f"  OCI layout: {artifact.path}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    layout = Path(args.layout)
    try:
        config, signature = oci.load_config_and_signature(layout)
    except (OSError, ValueError, KeyError) as exc:
        print(f"fleet verify: cannot read OCI layout {layout}: {exc}", file=sys.stderr)
        return 1
    manifest = PluginManifest.model_validate({**config, "signature": signature})
    if manifest.provenance is None or manifest.provenance.digest is None:
        print("fleet verify: manifest has no provenance digest to verify against", file=sys.stderr)
        return 1

    # Content integrity ("verify before trust", hub.md §2.3): every packaged blob must
    # hash to its content address, and the packaged SADF wire form must equal the signed
    # digest -- so a tampered *blob*, not just a tampered *claim*, is caught before load.
    try:
        oci.verify_asset_integrity(layout, manifest.provenance.digest)
    except (ValueError, KeyError) as exc:
        print(f"fleet verify: content integrity check failed: {exc}", file=sys.stderr)
        return 1

    trusted: bytes | None = None
    if args.pub:
        try:
            trusted = Path(args.pub).read_bytes()
        except OSError as exc:
            print(
                f"fleet verify: cannot read public key {args.pub}: {exc.strerror or exc}",
                file=sys.stderr,
            )
            return 1
    registry = PluginRegistry(
        require_signature=True, verifier=make_verifier(trusted_public_key_pem=trusted)
    )
    try:
        registry.register(manifest)
    except (RegistryError, SignatureError) as exc:
        print(f"fleet verify: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {manifest.name} v{manifest.version} signature + content verified; manifest loads")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    out = Path(args.output)
    assets_dir = Path(args.assets_dir) if args.assets_dir else out.parent / f"{out.stem}_assets"
    # Geometry refs must resolve relative to the document's directory.
    rel = os.path.relpath(str(assets_dir), str(out.parent or Path()))
    uri_prefix = rel.replace(os.sep, "/") + "/"
    try:
        doc = importers.import_description(
            args.path, assets_dir=assets_dir, uri_prefix=uri_prefix, fmt=args.format
        )
    except (importers.ImportError_, GeometryError, SadfError) as exc:
        print(f"fleet import: {exc}", file=sys.stderr)
        return 1
    if out.parent != Path():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_json(doc) + "\n", encoding="utf-8")
    print(f"imported {doc.asset.identity.id} -> {out}")
    print(f"  geometry: {len(doc.asset.geometry)} ref(s) under {assets_dir}")
    return 0


def _cmd_fidelity(args: argparse.Namespace) -> int:
    loaded = _load(args.path)
    if isinstance(loaded, Diagnostic):
        print(f"{loaded.source}: {loaded.message}", file=sys.stderr)
        return 1
    asset = loaded.asset
    try:
        declared = fidelity.profiles(asset)  # validated + ordered coarse -> fine
    except fidelity.FidelityError as exc:
        print(f"fleet fidelity: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {
            "identity": asset.identity.id,
            "profiles": [
                {
                    "tier": p.tier.value,
                    "determinism_class": p.determinism_class.value,
                    "detail": p.detail,
                }
                for p in declared
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"{asset.identity.id}: {len(declared)} fidelity profile(s)")
    for p in declared:
        detail = f" — {p.detail}" if p.detail else ""
        print(f"  {p.tier.value} [{p.determinism_class.value}]{detail}")
    if not declared:
        print("  (single-fidelity asset)")
    return 0


def _cmd_families(args: argparse.Namespace) -> int:
    from astro_mine.fleet.templates import FAMILIES

    if args.json:
        payload = {
            "families": [
                {
                    "name": fam.name,
                    "kind": fam.kind,
                    "summary": fam.summary,
                    "params": [
                        {
                            "name": s.name,
                            "min": s.minimum,
                            "max": s.maximum,
                            "default": s.default,
                            "unit": s.unit,
                        }
                        for s in fam.params
                    ],
                }
                for fam in FAMILIES.values()
            ]
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for fam in FAMILIES.values():
        print(f"{fam.name} [{fam.kind}] — {fam.summary}")
        for s in fam.params:
            print(f"  {s.name}: [{s.minimum}, {s.maximum}] {s.unit} (default {s.default})")
    return 0


def _cmd_resolve_family(args: argparse.Namespace) -> int:
    from astro_mine.fleet.params import ParamError
    from astro_mine.fleet.templates import resolve_family

    overrides: dict[str, float] = {}
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            print(f"fleet resolve-family: bad --set {item!r}; expected KEY=VALUE", file=sys.stderr)
            return 1
        try:
            overrides[key] = float(raw)
        except ValueError:
            print(
                f"fleet resolve-family: --set {key} value {raw!r} is not a number", file=sys.stderr
            )
            return 1
    try:
        doc = resolve_family(
            args.family, overrides, variant=args.variant, version=args.asset_version
        )
    except ParamError as exc:
        print(f"fleet resolve-family: {exc}", file=sys.stderr)
        return 1
    rendered = canonical_json(doc)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"resolved {doc.asset.identity.id} -> {args.output}")
    else:
        print(rendered)
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    # Publishing is always signed: hub.md §9 defines no namespace tier for unsigned content, so
    # Hub's admission gate refuses it (astro-mine-hub#32). `--sign` is therefore redundant on this
    # command and only `--key` is consulted; `fleet package` keeps optional signing, because a
    # local OCI artifact never reaches Hub.
    if not args.key:
        print(
            "fleet publish: --key <private-key.pem> is required; Hub admits no unsigned content "
            "(mint a key with `astro-mine-hub keygen`)",
            file=sys.stderr,
        )
        return 1
    loaded = _load(args.path)
    if isinstance(loaded, Diagnostic):
        print(f"{loaded.source}: {loaded.message}", file=sys.stderr)
        return 1

    try:
        sign_key = Path(args.key).read_bytes()
    except OSError as exc:
        print(f"fleet publish: cannot read key {args.key}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    from astro_mine.fleet.capabilities import CapabilityError
    from astro_mine.fleet.packaging.hub import HubError, publish_asset, pull_asset

    try:
        pub = publish_asset(
            loaded,
            args.registry,
            sign_key=sign_key,
            base_dir=Path(args.path).parent,
            namespace=args.namespace,
            publisher=args.publisher,
        )
    except (HubError, CapabilityError) as exc:
        print(f"fleet publish: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"fleet publish: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.pub:
        try:
            trusted = Path(args.pub).read_bytes()
        except OSError as exc:
            print(
                f"fleet publish: cannot read public key {args.pub}: {exc.strerror or exc}",
                file=sys.stderr,
            )
            return 1
        require = None if pub.signed else ()
        redoc = pull_asset(
            args.registry, pub.digest, trusted_public_key_pem=trusted, require=require
        )
        if redoc.asset.identity.id != loaded.asset.identity.id:  # pragma: no cover - defensive
            print("fleet publish: round-trip identity mismatch", file=sys.stderr)
            return 1

    if args.json:
        print(
            json.dumps(
                {
                    "reference": pub.reference,
                    "digest": pub.digest,
                    "asset_digest": pub.asset_digest,
                    "namespace": pub.namespace,
                    "signed": pub.signed,
                },
                sort_keys=True,
            )
        )
    else:
        suffix = " (signed)" if pub.signed else ""
        verified = " — round-trip verified" if args.pub else ""
        print(f"published {pub.reference} -> {pub.digest}{suffix}{verified}")
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    from astro_mine.fleet.capabilities import CapabilityError
    from astro_mine.fleet.catalog import asset_preview, list_menu, materialize_preview
    from astro_mine.fleet.packaging.hub import HubError

    if args.materialize is not None and args.preview is None:
        print("fleet catalog: --materialize requires --preview <reference>", file=sys.stderr)
        return 1

    if args.preview is not None and args.materialize is not None:
        try:
            document = materialize_preview(args.registry, args.preview, args.materialize)
        except HubError as exc:
            print(f"fleet catalog: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"fleet catalog: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {"reference": args.preview, "document": str(document)},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"materialized {args.preview} -> {document}")
        return 0

    if args.preview is not None:
        try:
            refs = asset_preview(args.registry, args.preview, fmt=args.format)
        except HubError as exc:
            print(f"fleet catalog: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"fleet catalog: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "reference": args.preview,
                        "format": args.format,
                        "geometry": [
                            {"role": r.role.value, "uri": r.uri, "frame": r.frame} for r in refs
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(f"{args.preview}: {len(refs)} {args.format} geometry ref(s)")
        for r in refs:
            print(f"  {r.role.value}: {r.uri} (frame {r.frame})")
        if not refs:
            print("  (no preview geometry — e.g. a mass-model asset)")
        return 0

    requires = [t for t in (args.requires or "").split(",") if t] or None
    try:
        entries = list_menu(args.registry, requires=requires)
    except (HubError, CapabilityError) as exc:
        print(f"fleet catalog: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"fleet catalog: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "registry": str(args.registry),
                    "assets": [
                        {
                            "reference": m.reference,
                            "digest": m.digest,
                            "kind": m.kind,
                            "namespace": m.namespace,
                            "capability_tags": list(m.capability_tags),
                        }
                        for m in entries
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"{len(entries)} asset(s) in {args.registry}")
    for m in entries:
        tags = ", ".join(m.capability_tags) or "(no capability tags)"
        print(f"  {m.reference} [{m.kind}] — {tags}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from astro_mine.fleet import exporters

    loaded = _load(args.path)
    if isinstance(loaded, Diagnostic):
        print(f"{loaded.source}: {loaded.message}", file=sys.stderr)
        return 1
    try:
        result = exporters.export_description(
            loaded,
            args.output,
            fmt=args.format,
            base_dir=Path(args.path).parent,
            assets_dir=args.assets_dir,
            fidelity=FidelityTier(args.fidelity),
        )
    except (exporters.ExportError, GeometryError) as exc:
        print(f"fleet export: {exc}", file=sys.stderr)
        return 1
    return _report_export(args, loaded, result, verb="exported")


def _cmd_render(args: argparse.Namespace) -> int:
    from astro_mine.fleet import exporters

    loaded = _load(args.path)
    if isinstance(loaded, Diagnostic):
        print(f"{loaded.source}: {loaded.message}", file=sys.stderr)
        return 1
    try:
        result = exporters.render_preview(
            loaded,
            args.output,
            base_dir=Path(args.path).parent,
            fmt=args.format,
            fidelity=FidelityTier(args.fidelity),
        )
    except (exporters.ExportError, GeometryError) as exc:
        print(f"fleet render: {exc}", file=sys.stderr)
        return 1
    return _report_export(args, loaded, result, verb="rendered")


def _report_export(
    args: argparse.Namespace,
    doc: SadfDocument,
    result: ExportResult,
    *,
    verb: str,
) -> int:
    """Print an export/render result and its **loss report** (fleet.md §11).

    Every export is lossy — no robot-description format holds a power budget or a sensor model —
    so the losses are always surfaced, never swallowed. They are informational: a lossy export is
    the expected outcome, so the exit code stays 0.
    """
    if args.json:
        print(
            json.dumps(
                {
                    "asset": doc.asset.identity.id,
                    "format": args.format,
                    "path": str(result.path),
                    "meshes": [str(p) for p in result.mesh_paths],
                    "losses": [
                        {"rule": loss.rule, "path": loss.path, "message": loss.message}
                        for loss in result.losses
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"{verb} {doc.asset.identity.id} -> {result.path}")
    if result.mesh_paths:
        print(f"  meshes: {len(result.mesh_paths)} file(s) under {result.mesh_paths[0].parent}")
    if result.losses:
        print(f"  lossy ({len(result.losses)}) — SADF stays authoritative:")
        for loss in result.losses:
            print(f"    [{loss.rule}] {loss.path}: {loss.message}", file=sys.stderr)
    return 0


# --- parser ----------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "new": _cmd_new,
    "validate": _cmd_validate,
    "lint": _cmd_lint,
    "resolve": _cmd_resolve,
    "package": _cmd_package,
    "verify": _cmd_verify,
    "publish": _cmd_publish,
    "catalog": _cmd_catalog,
    "import": _cmd_import,
    "fidelity": _cmd_fidelity,
    "families": _cmd_families,
    "resolve-family": _cmd_resolve_family,
    "export": _cmd_export,
    "render": _cmd_render,
}

#: The fidelity tiers a caller may dial an export/render to (the deferred ``surrogate`` tier is
#: not one of them — :mod:`astro_mine.fleet.fidelity` rejects it).
_FIDELITY_CHOICES = [tier.value for tier in fidelity.FIDELITY_ORDER]


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON diagnostics"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-mine-fleet",
        description="Author, validate, lint, package, sign, and verify SADF assets.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_new = sub.add_parser("new", help="scaffold a minimal, valid SADF asset")
    p_new.add_argument("kind", help="asset kind label (e.g. rover, orbiter, excavator)")
    p_new.add_argument("output", help="path to write the scaffold to")
    p_new.add_argument("--id", help="asset identity id (default: example.<kind>)")
    p_new.add_argument("--name", help="asset display name")
    p_new.add_argument("--asset-version", default="0.1.0", help="asset version (default: 0.1.0)")
    p_new.add_argument("--force", action="store_true", help="overwrite an existing file")

    p_validate = sub.add_parser("validate", help="validate one SADF document")
    p_validate.add_argument("path", help="path to a SADF document")
    _add_json_flag(p_validate)

    p_lint = sub.add_parser("lint", help="lint one or more SADF documents")
    p_lint.add_argument("paths", nargs="+", help="paths to SADF documents")
    _add_json_flag(p_lint)

    p_resolve = sub.add_parser("resolve", help="emit a document's canonical JSON form")
    p_resolve.add_argument("path", help="path to a SADF document")
    p_resolve.add_argument("-o", "--output", help="write to a file instead of stdout")

    p_package = sub.add_parser("package", help="write a content-addressed asset bundle")
    p_package.add_argument("path", help="path to a SADF document")
    p_package.add_argument(
        "--out", default=_DEFAULT_PACKAGE_OUT, help=f"output dir (default: {_DEFAULT_PACKAGE_OUT})"
    )
    p_package.add_argument(
        "--oci", action="store_true", help="emit a content-addressed OCI image layout"
    )
    p_package.add_argument(
        "--sign", action="store_true", help="sign the OCI artifact (requires --oci and --key)"
    )
    p_package.add_argument(
        "--key", help="ECDSA P-256 private-key PEM for --sign (see `astro-mine-hub keygen`)"
    )
    _add_json_flag(p_package)

    p_verify = sub.add_parser(
        "verify", help="verify an OCI asset artifact's signature and that it loads"
    )
    p_verify.add_argument("layout", help="path to an OCI image-layout directory")
    p_verify.add_argument("--pub", help="trusted ECDSA P-256 public-key PEM (pins the signer)")

    p_publish = sub.add_parser("publish", help="publish a signed SADF asset to a Hub registry")
    p_publish.add_argument("path", help="path to a SADF document")
    p_publish.add_argument(
        "--registry",
        required=True,
        help="local OCI-layout registry path, or a remote registry URL (e.g. ghcr.io/astro-mine)",
    )
    p_publish.add_argument(
        "--sign", action="store_true", help="cosign-sign the artifact (requires --key)"
    )
    p_publish.add_argument(
        "--key", help="ECDSA P-256 private-key PEM for --sign (see `astro-mine-hub keygen`)"
    )
    p_publish.add_argument(
        "--pub", help="trusted public-key PEM to re-verify the round-trip after publish"
    )
    p_publish.add_argument("--namespace", default="open", help="Hub namespace (default: open)")
    p_publish.add_argument("--publisher", default="local", help="publisher id (default: local)")
    _add_json_flag(p_publish)

    p_catalog = sub.add_parser(
        "catalog", help="list a Hub registry as the robot menu; preview one asset's geometry"
    )
    p_catalog.add_argument(
        "--registry",
        required=True,
        help="local OCI-layout registry path, or a remote registry URL (e.g. ghcr.io/astro-mine)",
    )
    p_catalog.add_argument(
        "--requires",
        metavar="TAG[,TAG...]",
        help="only assets declaring all these Core capability tags (Mind/Allocate negotiation)",
    )
    p_catalog.add_argument(
        "--preview",
        metavar="REFERENCE",
        help="print one asset's geometry refs (name:version or sha256:...) instead of the menu",
    )
    p_catalog.add_argument(
        "--format",
        choices=["gltf", "usd"],
        default="gltf",
        help="geometry format for --preview (default: gltf)",
    )
    p_catalog.add_argument(
        "--materialize",
        metavar="DIR",
        help="with --preview: write a servable SADF-JSON + geometry dir; print the documentUrl",
    )
    _add_json_flag(p_catalog)

    p_import = sub.add_parser(
        "import", help="import a URDF/SDF description into SADF + USD/glTF geometry"
    )
    p_import.add_argument("path", help="path to a URDF or SDF description")
    p_import.add_argument("-o", "--output", required=True, help="path to write the SADF document")
    p_import.add_argument(
        "--assets-dir", help="directory for generated USD/glTF geometry (default: <output>_assets)"
    )
    p_import.add_argument(
        "--format", choices=["urdf", "sdf"], help="override format detection (by extension)"
    )

    p_fidelity = sub.add_parser(
        "fidelity", help="list an asset's multi-fidelity profiles (coarse -> fine)"
    )
    p_fidelity.add_argument("path", help="path to a SADF document")
    _add_json_flag(p_fidelity)

    p_families = sub.add_parser("families", help="list the parametric asset families + parameters")
    _add_json_flag(p_families)

    p_resolve_family = sub.add_parser(
        "resolve-family", help="resolve a parametric family to a concrete SADF document"
    )
    p_resolve_family.add_argument("family", help="family handle (see `fleet families`)")
    p_resolve_family.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a parameter"
    )
    p_resolve_family.add_argument(
        "--variant", default="custom", help="variant name (identity suffix)"
    )
    p_resolve_family.add_argument(
        "--asset-version", default="0.1.0", help="asset version (default: 0.1.0)"
    )
    p_resolve_family.add_argument("-o", "--output", help="write to a file instead of stdout")

    p_export = sub.add_parser(
        "export", help="export a SADF asset to URDF/SDF (ROS) or a USD stage (Sim/Studio)"
    )
    p_export.add_argument("path", help="path to a SADF document")
    p_export.add_argument("-o", "--output", required=True, help="path to write the description to")
    p_export.add_argument(
        "--format",
        choices=list(exporters.FORMATS),
        default="urdf",
        help="target description format (default: urdf)",
    )
    p_export.add_argument(
        "--assets-dir",
        help="directory for the generated URDF/SDF meshes (default: <output>_meshes; "
        "USD references the asset's own geometry in place and ignores this)",
    )
    p_export.add_argument(
        "--fidelity",
        choices=_FIDELITY_CHOICES,
        default=FidelityTier.ARTICULATED.value,
        help="fidelity tier selecting the visual LOD (default: articulated — the finest mesh). "
        "URDF/SDF carry one tier; USD carries them all",
    )
    _add_json_flag(p_export)

    p_render = sub.add_parser(
        "render", help="render an asset preview/thumbnail (a composed, posed glTF/USD scene)"
    )
    p_render.add_argument("path", help="path to a SADF document")
    p_render.add_argument("-o", "--output", required=True, help="path to write the preview to")
    p_render.add_argument(
        "--format",
        choices=list(exporters.PREVIEW_FORMATS),
        default="glb",
        help="preview format (default: glb — the web/View form; usd for Sim/Studio)",
    )
    p_render.add_argument(
        "--fidelity",
        choices=_FIDELITY_CHOICES,
        default=FidelityTier.KINEMATIC.value,
        help="fidelity tier selecting the visual LOD (default: kinematic — a thumbnail has no "
        "use for the finest mesh)",
    )
    _add_json_flag(p_render)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``fleet`` console command.

    Parses ``argv`` (defaults to ``sys.argv``), dispatches to the subcommand handler,
    and exits with its status. A non-zero handler result raises ``SystemExit``;
    argparse itself exits non-zero on a usage error (e.g. no subcommand).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    code = _HANDLERS[args.command](args)
    if code:
        raise SystemExit(code)


def deprecated_alias(argv: Sequence[str] | None = None) -> None:
    """The pre-RFC-0011 ``fleet`` name — kept for one deprecation cycle.

    ``fleet`` was a generic binary planted on every user's ``PATH``; the platform now names a
    component's command after its package (``conventions.md §13``, normative). The old name keeps
    working unchanged, prints one line naming its replacement, and is **removed at the first
    public-benchmark milestone** — i.e. before the platform is public, so no outside user ever
    learns the transitional name.

    The notice goes to **stderr**, never stdout: `astro-mine-fleet --json`-style output has to stay
    machine-readable, and a warning on stdout would corrupt exactly the pipelines most likely to
    be using the old name in a script.
    """
    print(
        "warning: `fleet` is deprecated and will be removed at the first public-benchmark "
        "milestone; use `astro-mine-fleet` instead (RFC-0011 §5).",
        file=sys.stderr,
    )
    return main(argv)
