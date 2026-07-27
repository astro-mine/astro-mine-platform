"""The ``astro-mine-hub`` CLI (RM-P1-HUB-06) — keygen / publish / search / resolve / pull / verify.

A thin argparse front-end over :class:`~astro_mine.hub.client.HubClient`. ``--registry`` takes
**either** transport (hub.md §7): a local OCI-layout directory (``./reg`` — fully offline, no hosted
Hub) **or** a remote OCI registry (``ghcr.io/astro-mine``, ``http://localhost:5000`` — the
Distribution Spec, with standard Docker credentials), resolved by
:func:`~astro_mine.hub.registry.open_registry`. Discovery commands rebuild the catalog from the
registry, so the CLI is stateless over it.

``pull`` returns the **Core manifest** (the config blob) by default and the artifact's **verified
payload layers** with ``--payload`` — the bytes are re-hashed against the verified manifest inside
the client either way, so a tampered artifact fails closed with a non-zero exit.

**Bad input is an error, never a traceback.** Every command reports a user-input failure as one
``astro-mine-hub <verb>: <what went wrong>`` line on stderr with a non-zero exit — the platform's
standard (``astro-mine-bench score``'s typed refusal, ``bench submit``'s missing-token message). A
traceback tells a user their *input* was wrong by making it look like the *tool* broke. :func:`main`
carries the backstop so a new verb cannot reintroduce one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from astro_mine.core.registry import ManifestDocument, PluginManifest, validate_manifest
from astro_mine.core.registry.loader import ManifestValidationError
from astro_mine.hub.client._client import HubClient, catalog_from_registry
from astro_mine.hub.registry import (
    ARTIFACT_KINDS,
    ArtifactNotFound,
    Blob,
    IntegrityError,
    RegistryHttpError,
    open_registry,
)
from astro_mine.hub.resolve import ResolutionError, ResolutionRequest
from astro_mine.hub.search import SearchQuery, search
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair

__all__ = ["main"]

#: The `manifest_version` a bare :class:`PluginManifest` is wrapped with so both on-disk forms are
#: validated by one reader. Pinned to the schema minor `ManifestDocument` accepts.
_MANIFEST_VERSION = "0.1"


class InputError(Exception):
    """A problem with what the user passed in — reported as one line, never as a traceback."""


def _client(
    registry_location: str, *, with_catalog: bool = False, trusted_key: bytes | None = None
) -> HubClient:
    registry = open_registry(registry_location)
    catalog = catalog_from_registry(registry) if with_catalog else None
    return HubClient(registry, catalog=catalog, trusted_public_key_pem=trusted_key)


def _read_bytes(path: str, what: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {what} {path}: {exc.strerror or exc}") from exc


def _read_manifest(path: str) -> PluginManifest:
    """Read a Core plugin manifest in **either** on-disk form, YAML or JSON.

    Two readers of the same artifact used to disagree: `astro-mine-core validate` accepts a
    manifest **document** (``manifest_version`` + a ``manifest:`` mapping) — the form both shipped
    examples in `astro-mine-core/examples/plugins/` use — while this flag required a **bare**
    ``PluginManifest``, in strict JSON. So the examples the guide tells a reader to copy validated
    `OK` and then could not be published, and nothing in the error pointed at the wrapper.

    Both forms are now accepted, discriminated on the wrapper keys (``manifest`` is not a
    ``PluginManifest`` field and the schema forbids unknown ones, so the test is unambiguous), and a
    bare manifest is wrapped and validated through the **same** Core gate — so publish agrees with
    `validate` on structure *and* on the semantic checks it runs, including the gated-capability
    export-control gate.
    """
    text = _read_bytes(path, "manifest")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InputError(f"{path} is not readable as YAML or JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(
            f"{path}: a plugin manifest must be a YAML/JSON mapping — either a manifest document "
            f"(`manifest_version` + `manifest:`) or a bare PluginManifest"
        )
    document = (
        data
        if "manifest" in data or "manifest_version" in data
        else {"manifest_version": _MANIFEST_VERSION, "manifest": data}
    )
    try:
        validate_manifest(document)
    except ManifestValidationError as exc:
        raise InputError(f"{path} is not a valid plugin manifest: {exc}") from exc
    return ManifestDocument.model_validate(document).manifest


def _cmd_publish(args: argparse.Namespace) -> int:
    manifest = _read_manifest(args.manifest)
    key = _read_bytes(args.key, "signing key")
    layers = [
        Blob("application/octet-stream", _read_bytes(path, "payload layer"))
        for path in (args.layer or [])
    ]
    artifact = _client(args.registry).publish(
        name=args.name,
        version=args.version,
        kind=args.kind,
        manifest=manifest,
        layers=layers,
        private_key_pem=key,
    )
    print(artifact.digest)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    client = _client(args.registry, with_catalog=True)
    query = SearchQuery(text=args.text, semantic=args.semantic, kind=args.kind, limit=args.limit)
    for result in search(client.catalog, query):
        print(f"{result.entry.reference}\t{result.entry.digest}\t{result.score:.3f}")
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    client = _client(args.registry, with_catalog=True)
    try:
        resolution = client.resolve(ResolutionRequest(name=args.name, version_spec=args.spec or ""))
    except ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{resolution.primary.reference}\t{resolution.primary.digest}")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    key = _read_bytes(args.trusted_key, "trusted key") if args.trusted_key else None
    client = _client(args.registry, trusted_key=key)
    try:
        if args.payload:
            return _write_payload(client, args)
        config = client.pull(args.reference, verify=not args.no_verify)
    except (SupplyChainError, IntegrityError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    except (ArtifactNotFound, RegistryHttpError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_bytes(config)
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(config)
    return 0


def _write_payload(client: HubClient, args: argparse.Namespace) -> int:
    """``pull --payload``: verified layers → a content-addressed dir, or stdout if single."""
    verify = not args.no_verify
    if args.out:
        for path in client.materialize(args.reference, dest=args.out, verify=verify):
            print(path)
        return 0
    layers = client.pull_payload(args.reference, verify=verify)
    if len(layers) != 1:
        print(
            f"error: {args.reference} has {len(layers)} payload layers; "
            f"use --out DIR to materialize them",
            file=sys.stderr,
        )
        return 1
    sys.stdout.buffer.write(layers[0].data)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    key = _read_bytes(args.trusted_key, "trusted key") if args.trusted_key else None
    try:
        digest = _client(args.registry, trusted_key=key).verify(args.reference)
    except (SupplyChainError, IntegrityError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    except (ArtifactNotFound, RegistryHttpError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ok {digest}")
    return 0


_REGISTRY_HELP = (
    "a local OCI-layout directory (./reg) or a remote OCI registry "
    "(ghcr.io/astro-mine, http://localhost:5000)"
)


def _cmd_keygen(args: argparse.Namespace) -> int:
    """Write a fresh cosign ECDSA-P256 keypair to ``cosign.key`` / ``cosign.pub``.

    The one signing-key command for the platform — the key `hub publish --key` signs with, and whose
    public half consumers pin as their trust anchor (`hub verify --trusted-key`). Seal owns the
    primitive (`generate_keypair`); Hub, the supply-chain tool, is where it belongs on the CLI.
    """
    private_pem, public_pem = generate_keypair()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cosign.key").write_bytes(private_pem)
    (out / "cosign.pub").write_bytes(public_pem)
    print(f"wrote {out / 'cosign.key'} and {out / 'cosign.pub'}")
    return 0


# --- Per-verb argument sets -------------------------------------------------------------------
#
# Each verb's flags live in one function so they can be attached to *either* parser: this package's
# own `astro-mine-hub <verb>`, and the umbrella's `astro-mine <verb>` (RFC-0011 §3, wired in
# astro_mine.hub.umbrella). Declaring them once is what stops the two surfaces from drifting.
#
# `resolve` and `keygen` are extracted for symmetry but are NOT umbrella verbs: RFC-0011 §2 puts
# publish/search/pull/verify at the top level, and `resolve` (a Hub-internal pin lookup) and
# `keygen` (key material) read better as Hub-scoped commands than as platform verbs.


def add_publish_arguments(parser: argparse.ArgumentParser) -> None:
    """`publish` — publish + sign an artifact."""
    parser.add_argument("--registry", required=True, help=_REGISTRY_HELP)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--kind", required=True, choices=ARTIFACT_KINDS)
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Core plugin manifest file — a manifest document (`manifest_version` + `manifest:`, "
            "what `astro-mine-core validate` and the shipped examples use) or a bare "
            "PluginManifest; YAML or JSON either way"
        ),
    )
    parser.add_argument(
        "--key",
        required=True,
        help="ECDSA private-key PEM to sign with (mint one with `astro-mine-hub keygen`)",
    )
    parser.add_argument("--layer", action="append", help="payload layer file (repeatable)")


def add_search_arguments(parser: argparse.ArgumentParser) -> None:
    """`search` — discover artifacts."""
    parser.add_argument("--registry", required=True, help=_REGISTRY_HELP)
    parser.add_argument("--text")
    parser.add_argument("--semantic")
    # Validated against the same closed set as `publish --kind`. A free-form value here silently
    # matched nothing, which reads as "no results" rather than "that is not a kind" — the two are
    # very different answers to a search.
    parser.add_argument("--kind", choices=ARTIFACT_KINDS)
    parser.add_argument("--limit", type=int, default=20)


def add_resolve_arguments(parser: argparse.ArgumentParser) -> None:
    """`resolve` — resolve to a pinned digest (Hub-scoped; not an umbrella verb)."""
    parser.add_argument("--registry", required=True, help=_REGISTRY_HELP)
    parser.add_argument("--name", required=True)
    parser.add_argument("--spec", help="PEP 440 version specifier")


def add_pull_arguments(parser: argparse.ArgumentParser) -> None:
    """`pull` — pull + re-verify an artifact."""
    parser.add_argument("--registry", required=True, help=_REGISTRY_HELP)
    parser.add_argument("reference")
    parser.add_argument("--out", help="write to this file (or, with --payload, this directory)")
    parser.add_argument(
        "--payload",
        action="store_true",
        help="pull the verified payload layers instead of the Core manifest",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--trusted-key", help="pin the signer to this public-key PEM")


def add_verify_arguments(parser: argparse.ArgumentParser) -> None:
    """`verify` — re-verify an artifact's supply chain."""
    parser.add_argument("--registry", required=True, help=_REGISTRY_HELP)
    parser.add_argument("reference")
    parser.add_argument("--trusted-key")


def add_keygen_arguments(parser: argparse.ArgumentParser) -> None:
    """`keygen` — mint a cosign keypair (Hub-scoped; not an umbrella verb)."""
    parser.add_argument("--out", required=True, help="directory to write cosign.key / cosign.pub")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astro-mine-hub", description="Astro-Mine Hub client")
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser("publish", help="publish + sign an artifact")
    add_publish_arguments(publish)
    publish.set_defaults(func=_cmd_publish)

    do_search = sub.add_parser("search", help="discover artifacts")
    add_search_arguments(do_search)
    do_search.set_defaults(func=_cmd_search)

    do_resolve = sub.add_parser("resolve", help="resolve to a pinned digest")
    add_resolve_arguments(do_resolve)
    do_resolve.set_defaults(func=_cmd_resolve)

    pull = sub.add_parser("pull", help="pull + re-verify an artifact")
    add_pull_arguments(pull)
    pull.set_defaults(func=_cmd_pull)

    verify = sub.add_parser("verify", help="re-verify an artifact")
    add_verify_arguments(verify)
    verify.set_defaults(func=_cmd_verify)

    keygen = sub.add_parser("keygen", help="generate a cosign ECDSA-P256 signing keypair")
    add_keygen_arguments(keygen)
    keygen.set_defaults(func=_cmd_keygen)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected command; return its exit code.

    The backstop for the no-tracebacks-on-user-input rule. The verbs report what they can name
    precisely; this catches the classes that are always the user's input rather than a Hub bug — an
    unreadable path, a manifest that does not validate, a registry that answers with an error — so a
    new verb cannot reintroduce a traceback by forgetting a ``try``. Anything else still raises,
    because a real defect must not be dressed up as bad input.
    """
    args = _parser().parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except (
        InputError,
        ManifestValidationError,
        OSError,
        ArtifactNotFound,
        RegistryHttpError,
        IntegrityError,
        SupplyChainError,
        ResolutionError,
    ) as exc:
        print(f"astro-mine-hub {args.command}: {exc}", file=sys.stderr)
        return 1
    return exit_code
