"""CLI: publish, families, resolve-family (RM-P1-FLEET-10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.fleet import cli

from .conftest import VALID_SADF


def run(*argv: str) -> int:
    try:
        cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _asset(tmp_path: Path) -> Path:
    path = tmp_path / "asset.sadf.yaml"
    path.write_text(VALID_SADF, encoding="utf-8")
    return path


# --- families --------------------------------------------------------------------


def test_families_text_lists_every_family(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("families") == 0
    out = capsys.readouterr().out
    for name in (
        "orbital-relay",
        "surface-rover",
        "manipulation-excavator",
        "logistics-hauler",
        "isru-plant",
    ):
        assert name in out


def test_families_json_carries_parameter_ranges(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("families", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    names = {fam["name"] for fam in payload["families"]}
    assert "isru-plant" in names
    rover = next(f for f in payload["families"] if f["name"] == "surface-rover")
    masses = next(p for p in rover["params"] if p["name"] == "chassis_mass_kg")
    assert masses["min"] == 10.0 and masses["max"] == 500.0


# --- resolve-family --------------------------------------------------------------


def test_resolve_family_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("resolve-family", "surface-rover", "--set", "chassis_mass_kg=250") == 0
    assert "astro-mine.fleet.surface-rover" in capsys.readouterr().out


def test_resolve_family_to_file_is_valid_sadf(tmp_path: Path) -> None:
    out = tmp_path / "rover.sadf.json"
    assert run("resolve-family", "orbital-relay", "--variant", "big", "-o", str(out)) == 0
    assert out.is_file()
    assert run("validate", str(out)) == 0  # the resolved family is valid SADF


def test_resolve_family_rejects_malformed_set(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("resolve-family", "surface-rover", "--set", "oops") == 1
    assert "KEY=VALUE" in capsys.readouterr().err


def test_resolve_family_rejects_non_numeric_value(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("resolve-family", "surface-rover", "--set", "chassis_mass_kg=heavy") == 1
    assert "is not a number" in capsys.readouterr().err


def test_resolve_family_rejects_out_of_range(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("resolve-family", "surface-rover", "--set", "chassis_mass_kg=5") == 1
    assert "resolve-family" in capsys.readouterr().err


# --- publish ---------------------------------------------------------------------


def test_publish_signed_with_roundtrip_verification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_mine.hub.supply_chain import generate_keypair

    src = _asset(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    priv, public = generate_keypair()  # signing key: `astro-mine-hub keygen`, minted directly here
    (keys / "asset-signing.key").write_bytes(priv)
    (keys / "asset-signing.pub").write_bytes(public)
    reg = tmp_path / "reg"
    code = run(
        "publish",
        str(src),
        "--registry",
        str(reg),
        "--sign",
        "--key",
        str(keys / "asset-signing.key"),
        "--pub",
        str(keys / "asset-signing.pub"),
        "--json",
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signed"] is True
    assert payload["digest"].startswith("sha256:")
    assert payload["reference"] == "test.rover:0.1.0"


def test_publish_without_a_key_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`fleet publish` always signs — Hub admits no unsigned content (astro-mine-hub#32).

    `fleet package` keeps optional signing: a local OCI artifact never reaches Hub."""
    src = _asset(tmp_path)
    reg = tmp_path / "reg"
    assert run("publish", str(src), "--registry", str(reg)) == 1
    assert "--key" in capsys.readouterr().err
    assert not reg.exists()


def test_publish_sign_without_key_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _asset(tmp_path)
    assert run("publish", str(src), "--registry", str(tmp_path / "reg"), "--sign") == 1
    assert "--key" in capsys.readouterr().err


def test_publish_immutable_republish_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_mine.hub.supply_chain import generate_keypair

    src = _asset(tmp_path)
    reg = tmp_path / "reg"
    priv, _ = generate_keypair()
    key = tmp_path / "cosign.key"
    key.write_bytes(priv)
    argv = ("publish", str(src), "--registry", str(reg), "--key", str(key))
    assert run(*argv) == 0
    capsys.readouterr()
    assert run(*argv) == 1  # name:version already exists
    assert "fleet publish" in capsys.readouterr().err


def test_publish_invalid_sadf_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "broken.sadf.yaml"
    bad.write_text("sadf_version: '0.1'\nasset:\n  identity:\n    id: x\n", encoding="utf-8")
    assert run("publish", str(bad), "--registry", str(tmp_path / "reg")) == 1
    assert capsys.readouterr().err
