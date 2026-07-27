"""The `package --oci --sign` / `verify` CLI flow (RM-P0-FLEET-06).

The signing key comes from `astro-mine-hub keygen` (the one signing-key command); here it is minted
directly from the same `generate_keypair` primitive rather than through another package's CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.fleet import cli


def run(*argv: str) -> int:
    try:
        cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    from astro_mine.hub.supply_chain import generate_keypair

    keys = tmp_path / "keys"
    keys.mkdir(exist_ok=True)
    private_pem, public_pem = generate_keypair()
    key, pub = keys / "asset-signing.key", keys / "asset-signing.pub"
    key.write_bytes(private_pem)
    pub.write_bytes(public_pem)
    return key, pub


def test_sign_and_verify_roundtrip(tmp_path: Path, valid_file: Path) -> None:
    private, public = _keys(tmp_path)
    layout = tmp_path / "oci"
    assert (
        run(
            "package",
            str(valid_file),
            "--out",
            str(layout),
            "--oci",
            "--sign",
            "--key",
            str(private),
        )
        == 0
    )
    assert (layout / "oci-layout").exists() and (layout / "index.json").exists()
    assert run("verify", str(layout), "--pub", str(public)) == 0
    assert run("verify", str(layout)) == 0  # embedded-cert (dev) path


def test_package_oci_json_output(
    tmp_path: Path, valid_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = tmp_path / "oci"
    assert run("package", str(valid_file), "--out", str(layout), "--oci", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["digest"].startswith("sha256:") and payload["signed"] is False


def test_plain_bundle_still_works(
    tmp_path: Path, valid_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bundle"
    assert run("package", str(valid_file), "--out", str(out)) == 0
    assert "sha256:" in capsys.readouterr().out


def test_verify_refuses_an_unsigned_artifact(tmp_path: Path, valid_file: Path) -> None:
    layout = tmp_path / "oci"
    assert run("package", str(valid_file), "--out", str(layout), "--oci") == 0
    assert run("verify", str(layout)) == 1


def test_sign_requires_oci(valid_file: Path, tmp_path: Path) -> None:
    private, _ = _keys(tmp_path)
    assert run("package", str(valid_file), "--sign", "--key", str(private)) == 1


def test_sign_requires_a_key(tmp_path: Path, valid_file: Path) -> None:
    assert run("package", str(valid_file), "--out", str(tmp_path / "oci"), "--oci", "--sign") == 1


def test_sign_with_a_missing_key_file(tmp_path: Path, valid_file: Path) -> None:
    layout = tmp_path / "oci"
    assert (
        run(
            "package",
            str(valid_file),
            "--out",
            str(layout),
            "--oci",
            "--sign",
            "--key",
            str(tmp_path / "nope.key"),
        )
        == 1
    )


def test_verify_bad_layout(tmp_path: Path) -> None:
    assert run("verify", str(tmp_path / "does-not-exist")) == 1


def test_verify_missing_pub_file(tmp_path: Path, valid_file: Path) -> None:
    private, _ = _keys(tmp_path)
    layout = tmp_path / "oci"
    assert (
        run(
            "package",
            str(valid_file),
            "--out",
            str(layout),
            "--oci",
            "--sign",
            "--key",
            str(private),
        )
        == 0
    )
    assert run("verify", str(layout), "--pub", str(tmp_path / "nope.pub")) == 1
