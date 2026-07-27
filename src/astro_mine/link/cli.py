"""The ``link`` CLI — publish a contact plan to a Hub registry.

The operator-facing half of the Link→Hub path (link.md §6): ``link publish`` pushes a serialized
:class:`~astro_mine.core.messages.ContactPlan` to a **local OCI-layout** registry (or a remote one,
e.g. ``ghcr.io/astro-mine``) as a signed ``comms_model`` artifact and prints its content digest —
the value a Bench ``ScenarioSpec`` pins. The cosign ECDSA-P256 key that signs it comes from
``astro-mine-hub keygen`` — the one signing-key command. Offline by default: no hosted Hub, no Cloud
(hub.md principle 7; ``LUNAR-TR-004``).

The plan itself is produced by the library (or by ``scripts/build_anchor_contact_plan.py`` for the
anchor scenario) and handed to the CLI in Core's byte-stable wire form, so the CLI never re-derives
geometry — it only publishes bytes it was given.

Backlog: RM-P0-LINK-04 -- https://github.com/astro-mine/astro-mine-link/issues/25
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from astro_mine.core.messages import contact_plan_from_wire
from astro_mine.link.registry import publish_contact_plan

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """The ``link`` CLI entry point."""
    parser = argparse.ArgumentParser(prog="astro-mine-link", description="Astro-Mine-Link tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser(
        "publish", help="Publish a ContactPlan to a local OCI-layout Hub registry."
    )
    publish.add_argument("plan", type=Path, help="ContactPlan in Core wire form (.pb).")
    publish.add_argument(
        "--registry",
        required=True,
        help="Local OCI-layout registry path, or a remote registry URL (e.g. ghcr.io/astro-mine).",
    )
    publish.add_argument("--name", required=True, help="Artifact name (the stable content id).")
    publish.add_argument("--version", required=True, help="Artifact version (SemVer).")
    publish.add_argument(
        "--scenario-id", required=True, help="The scenario this comms model belongs to."
    )
    publish.add_argument(
        "--key",
        type=Path,
        required=True,
        help=(
            "Cosign ECDSA-P256 private key (PEM); signs the artifact. Required — Hub admits no "
            "unsigned content. Mint one with `astro-mine-hub keygen`."
        ),
    )
    publish.add_argument(
        "--input-hashes",
        type=Path,
        default=None,
        help="JSON object of pinned-input digests (kernels/terrain/nodes/epoch/config).",
    )
    publish.set_defaults(func=_publish)

    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


def _publish(args: argparse.Namespace) -> int:
    plan = contact_plan_from_wire(Path(args.plan).read_bytes())
    input_hashes: dict[str, str] = {}
    if args.input_hashes is not None:
        loaded: Any = json.loads(Path(args.input_hashes).read_text())
        input_hashes = {str(k): str(v) for k, v in dict(loaded).items()}
    private_key_pem = Path(args.key).read_bytes()

    artifact = publish_contact_plan(
        plan,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        scenario_id=args.scenario_id,
        input_hashes=input_hashes or None,
        private_key_pem=private_key_pem,
    )
    print(f"published {artifact.reference} -> {artifact.digest}")
    return 0


def deprecated_alias(argv: Sequence[str] | None = None) -> int:
    """The pre-RFC-0011 ``link`` name — kept for one deprecation cycle.

    ``link`` was a generic binary planted on every user's ``PATH``; the platform now names a
    component's command after its package (``conventions.md §13``, normative). The old name keeps
    working unchanged, prints one line naming its replacement, and is **removed at the first
    public-benchmark milestone** — i.e. before the platform is public, so no outside user ever
    learns the transitional name.

    The notice goes to **stderr**, never stdout: `astro-mine-link --json`-style output has to stay
    machine-readable, and a warning on stdout would corrupt exactly the pipelines most likely to
    be using the old name in a script.
    """
    print(
        "warning: `link` is deprecated and will be removed at the first public-benchmark "
        "milestone; use `astro-mine-link` instead (RFC-0011 §5).",
        file=sys.stderr,
    )
    return main(argv)
