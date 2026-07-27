"""CLI: catalog listing + geometry preview (RM-P1-FLEET-11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.core.sadf import SadfDocument
from astro_mine.fleet import cli
from astro_mine.fleet.library import load_reference
from astro_mine.fleet.packaging.hub import publish_asset
from astro_mine.hub.supply_chain import generate_keypair

from .test_catalog import _novel_geometry_asset, _publish_geometry_asset


def run(*argv: str) -> int:
    try:
        cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _signed_registry(tmp_path: Path, docs: list[SadfDocument]) -> Path:
    """Sign+publish ``docs`` to a fresh registry so `catalog --preview` verifies before trust."""
    private_pem, _ = generate_keypair()
    reg = tmp_path / "reg"
    for doc in docs:
        publish_asset(doc, reg, sign_key=private_pem)
    return reg


def test_catalog_json_lists_published_assets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    orbiter, excavator = load_reference("relay_orbiter"), load_reference("excavator")
    reg = _signed_registry(tmp_path, [orbiter, excavator])
    assert run("catalog", "--registry", str(reg), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    by_ref = {a["reference"]: a for a in payload["assets"]}
    # Read each asset's version off the document rather than restating it: the catalog's job is to
    # echo back the `id:version` it was published under, and hard-coding the version here would
    # turn every legitimate asset revision into a catalog-test failure.
    orbiter_ref = f"astro-mine.fleet.relay-orbiter:{orbiter.asset.identity.version}"
    excavator_ref = f"astro-mine.fleet.excavator:{excavator.asset.identity.version}"
    assert set(by_ref) == {orbiter_ref, excavator_ref}
    assert by_ref[orbiter_ref]["kind"] == "orbiter"
    assert "comms.relay" in by_ref[orbiter_ref]["capability_tags"]


def test_catalog_text_shows_vehicle_kind_and_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(tmp_path, [load_reference("relay_orbiter")])
    assert run("catalog", "--registry", str(reg)) == 0
    out = capsys.readouterr().out
    assert "[orbiter]" in out and "comms.relay" in out


def test_catalog_requires_filters_the_menu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(
        tmp_path, [load_reference("relay_orbiter"), load_reference("prospecting_rover")]
    )
    assert run("catalog", "--registry", str(reg), "--requires", "mobility.wheeled", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert [a["kind"] for a in payload["assets"]] == ["rover"]  # orbiter is filtered out


def test_catalog_requires_unknown_tag_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(tmp_path, [load_reference("relay_orbiter")])
    assert run("catalog", "--registry", str(reg), "--requires", "not.a.real.tag") == 1
    assert "fleet catalog" in capsys.readouterr().err


def test_catalog_preview_resolves_geometry_by_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(tmp_path, [_novel_geometry_asset()])
    ref = "example.hopper-mk1:0.1.0"

    assert run("catalog", "--registry", str(reg), "--preview", ref, "--json") == 0
    gltf = json.loads(capsys.readouterr().out)
    assert gltf["format"] == "gltf"
    assert [g["uri"] for g in gltf["geometry"]] == ["hopper.glb"]

    argv = ("catalog", "--registry", str(reg), "--preview", ref, "--format", "usd", "--json")
    assert run(*argv) == 0
    usd = json.loads(capsys.readouterr().out)
    assert [g["uri"] for g in usd["geometry"]] == ["hopper.usda"]


def test_catalog_preview_of_mass_model_reports_no_geometry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(tmp_path, [load_reference("relay_orbiter")])
    ref = "astro-mine.fleet.relay-orbiter:0.1.0"
    assert run("catalog", "--registry", str(reg), "--preview", ref) == 0
    assert "no preview geometry" in capsys.readouterr().out


def test_catalog_preview_of_unknown_reference_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(tmp_path, [load_reference("relay_orbiter")])
    assert run("catalog", "--registry", str(reg), "--preview", "no.such.asset:9.9.9") == 1
    assert capsys.readouterr().err


def test_catalog_materialize_writes_a_servable_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = tmp_path / "reg"
    ref = _publish_geometry_asset(reg, tmp_path / "src")
    out = tmp_path / "served"

    argv = (
        "catalog",
        "--registry",
        str(reg),
        "--preview",
        ref,
        "--materialize",
        str(out),
        "--json",
    )
    assert run(*argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["document"]).is_file()
    assert (out / "geometry" / "hopper.glb").read_bytes() == b"GLB-BYTES-123"


def test_catalog_materialize_requires_preview(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg = _signed_registry(tmp_path, [load_reference("relay_orbiter")])
    assert run("catalog", "--registry", str(reg), "--materialize", str(tmp_path / "x")) == 1
    assert "requires --preview" in capsys.readouterr().err
