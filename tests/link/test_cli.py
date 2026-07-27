"""RM-P0-LINK-04 — the ``link`` CLI: publish a ContactPlan to a local Hub registry.

The operator path (link.md §6): mint a cosign keypair, hand the CLI a plan in Core's byte-stable
wire form, and get back a signed, digest-resolvable ``comms_model`` artifact in a **local
OCI-layout** registry — offline, no hosted Hub (``LUNAR-TR-004``). This is the command
``scripts/build_anchor_contact_plan.py`` and a Bench maintainer drive to mint an anchor pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.core.messages import (
    ContactInterval,
    ContactNode,
    ContactPlan,
    contact_plan_to_wire,
)
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Registry
from astro_mine.link.cli import main

_NAME = "astro-mine.link.cli-test"
_VERSION = "0.1.0"
_SCENARIO = "cli-test-scenario"


def _plan_file(tmp_path: Path) -> Path:
    plan = ContactPlan(
        nodes=[
            ContactNode(id="rover", role=NodeRole.SPACE, kind="surface_agent"),
            ContactNode(id="dss", role=NodeRole.GROUND, kind="ground_station"),
        ],
        intervals=[ContactInterval(node_a="rover", node_b="dss", start_tdb_s=0.0, end_tdb_s=600.0)],
    )
    path = tmp_path / "plan.pb"
    path.write_bytes(contact_plan_to_wire(plan))
    return path


def test_publishes_a_signed_contact_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_mine.hub.supply_chain import generate_keypair

    keys = tmp_path / "keys"
    keys.mkdir()
    priv, public = generate_keypair()  # signing key: `astro-mine-hub keygen`, minted directly here
    (keys / "cosign.key").write_bytes(priv)
    (keys / "cosign.pub").write_bytes(public)

    hashes = tmp_path / "inputs.json"
    hashes.write_text(json.dumps({"kernels": "sha256:" + "ab" * 32}))
    registry = tmp_path / "registry"

    code = main(
        [
            "publish",
            str(_plan_file(tmp_path)),
            "--registry",
            str(registry),
            "--name",
            _NAME,
            "--version",
            _VERSION,
            "--scenario-id",
            _SCENARIO,
            "--key",
            str(keys / "cosign.key"),
            "--input-hashes",
            str(hashes),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert f"published {_NAME}:{_VERSION}" in out

    # The artifact the CLI wrote is a signed, verifiable comms_model resolvable by its digest.
    public_pem = (keys / "cosign.pub").read_bytes()
    client = HubClient(Registry(registry), trusted_public_key_pem=public_pem)
    manifest = PluginManifest.model_validate_json(client.pull(f"{_NAME}:{_VERSION}"))
    assert manifest.kind == PluginKind.COMMS_MODEL
    assert manifest.attributes["scenario_id"] == _SCENARIO


def test_publish_requires_a_signing_key(tmp_path: Path) -> None:
    """Unsigned publishing is refused, and the CLI says so before any work happens.

    `hub.md` §9 tiers artifacts as *open* (self-published, **signed**, unreviewed), *curated*, and
    *verified* — there is no tier for unsigned content, and Hub's admission gate refuses it
    (astro-mine-hub#32). Previously this stored the artifact with no attestations, leaving a
    consumer to pull it with an empty requirement set."""
    registry = tmp_path / "registry"
    with pytest.raises(SystemExit) as exit_info:  # argparse rejects the missing --key
        main(
            [
                "publish",
                str(_plan_file(tmp_path)),
                "--registry",
                str(registry),
                "--name",
                _NAME,
                "--version",
                _VERSION,
                "--scenario-id",
                _SCENARIO,
            ]
        )
    assert exit_info.value.code == 2
    assert not registry.exists()  # nothing was written


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
