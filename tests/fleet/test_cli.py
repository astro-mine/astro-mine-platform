"""The ``fleet`` CLI (RM-P0-FLEET-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.fleet import __version__, cli

from .conftest import INVALID_SADF, VALID_SADF


def run(*argv: str) -> int:
    """Invoke the CLI, returning the process exit code (0 when it does not exit)."""
    try:
        cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


# --- parser-level ----------------------------------------------------------------


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--version") == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_is_a_usage_error() -> None:
    assert run() == 2


def test_main_returns_none_on_success(valid_file: Path) -> None:
    assert cli.main(["validate", str(valid_file)]) is None


# --- new -------------------------------------------------------------------------


def test_new_scaffold_round_trips_through_validate(tmp_path: Path) -> None:
    out = tmp_path / "rover.sadf.yaml"
    assert run("new", "rover", str(out)) == 0
    assert out.exists()
    assert run("validate", str(out)) == 0  # the scaffold is valid by construction


def test_new_honors_id_name_and_version(tmp_path: Path) -> None:
    out = tmp_path / "a.yaml"
    assert (
        run(
            "new",
            "orbiter",
            str(out),
            "--id",
            "relay.sat",
            "--name",
            "Relay",
            "--asset-version",
            "2.0.0",
        )
        == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "relay.sat" in text and "Relay" in text and "2.0.0" in text


def test_new_refuses_to_overwrite_without_force(valid_file: Path) -> None:
    assert run("new", "rover", str(valid_file)) == 1


def test_new_force_overwrites(valid_file: Path) -> None:
    assert run("new", "rover", str(valid_file), "--force") == 0
    assert "example.rover" in valid_file.read_text(encoding="utf-8")


def test_new_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deep" / "a.yaml"
    assert run("new", "rover", str(out)) == 0
    assert out.exists()


def test_new_with_bare_filename_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run("new", "rover", "bare.yaml") == 0
    assert (tmp_path / "bare.yaml").exists()


# --- validate --------------------------------------------------------------------


def test_validate_ok(valid_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", str(valid_file)) == 0
    assert "valid SADF" in capsys.readouterr().out


def test_validate_ok_json(valid_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", str(valid_file), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "diagnostics": []}


def test_validate_rejects_invalid(invalid_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", str(invalid_file)) == 1
    assert str(invalid_file) in capsys.readouterr().err


def test_validate_invalid_json(invalid_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", str(invalid_file), "--json") == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["source"] == str(invalid_file)


def test_validate_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", str(tmp_path / "nope.yaml")) == 1
    assert "cannot read file" in capsys.readouterr().err


# --- lint ------------------------------------------------------------------------


def test_lint_passes_multiple(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(VALID_SADF, encoding="utf-8")
    b.write_text(VALID_SADF.replace("test.rover", "test.hauler"), encoding="utf-8")
    assert run("lint", str(a), str(b)) == 0
    assert "2 file(s) passed" in capsys.readouterr().out


def test_lint_reports_one_bad_file(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.yaml", tmp_path / "bad.yaml"
    good.write_text(VALID_SADF, encoding="utf-8")
    bad.write_text(INVALID_SADF, encoding="utf-8")
    assert run("lint", str(good), str(bad)) == 1


def test_lint_json_aggregates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(INVALID_SADF, encoding="utf-8")
    assert run("lint", str(bad), "--json") == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and len(payload["diagnostics"]) == 1


# --- resolve ---------------------------------------------------------------------


def test_resolve_to_stdout(valid_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("resolve", str(valid_file)) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["asset"]["identity"]["id"] == "test.rover"


def test_resolve_to_file(valid_file: Path, tmp_path: Path) -> None:
    dest = tmp_path / "canonical.json"
    assert run("resolve", str(valid_file), "-o", str(dest)) == 0
    assert json.loads(dest.read_text(encoding="utf-8"))["sadf_version"] == "0.1"


def test_resolve_load_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("resolve", str(tmp_path / "nope.yaml")) == 1
    assert "cannot read file" in capsys.readouterr().err


# --- package ---------------------------------------------------------------------


def test_package_text(valid_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("package", str(valid_file), "--out", str(tmp_path / "out")) == 0
    assert "sha256:" in capsys.readouterr().out


def test_package_json(valid_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("package", str(valid_file), "--out", str(tmp_path / "out"), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["digest"].startswith("sha256:")
    assert Path(payload["path"]).is_dir()


def test_package_load_error(invalid_file: Path) -> None:
    assert run("package", str(invalid_file)) == 1


# --- usage -----------------------------------------------------------------------


@pytest.mark.parametrize("command", ["import", "export", "render"])
def test_output_is_required(
    command: str, valid_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """These verbs write a file; argparse rejects the call that does not say where."""
    assert run(command, str(valid_file)) == 2
    assert f"fleet {command}" in capsys.readouterr().err


# --- export / render (RM-P0-FLEET-01/02) -----------------------------------------


@pytest.fixture
def rover_file(tmp_path: Path) -> Path:
    """The library prospecting rover, on disk — the CLI's "representative SADF asset"."""
    from astro_mine.fleet import library
    from astro_mine.fleet._core import canonical_json

    path = tmp_path / "rover.sadf.json"
    path.write_text(canonical_json(library.load_reference("prospecting_rover")), encoding="utf-8")
    return path


@pytest.mark.parametrize("fmt", ["urdf", "sdf", "usd"])
def test_export_writes_a_description(
    fmt: str, rover_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The acceptance criteria: `fleet export` produces URDF, SDF, and a USD stage — no stub."""
    out = tmp_path / f"out/rover.{fmt}"
    assert run("export", str(rover_file), "-o", str(out), "--format", fmt) == 0
    assert out.is_file() and out.stat().st_size > 0
    assert "exported astro-mine.fleet.prospecting-rover" in capsys.readouterr().out


def test_export_reports_its_losses_as_json(
    rover_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every export is lossy; `--json` makes the losses machine-readable (fleet.md §10, §11)."""
    out = tmp_path / "rover.urdf"
    assert run("export", str(rover_file), "-o", str(out), "--json") == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["asset"] == "astro-mine.fleet.prospecting-rover"
    assert payload["path"] == str(out)
    rules = {loss["rule"] for loss in payload["losses"]}
    assert "asset.block_dropped" in rules  # the power/thermal/sensor blocks URDF cannot hold
    assert all({"rule", "path", "message"} <= set(loss) for loss in payload["losses"])


def test_export_reports_losses_on_stderr_but_still_succeeds(
    rover_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lossy export is the *expected* outcome — it is reported, not treated as a failure."""
    assert run("export", str(rover_file), "-o", str(tmp_path / "r.urdf")) == 0
    captured = capsys.readouterr()
    assert "lossy" in captured.out
    assert "[asset.block_dropped]" in captured.err


def test_export_fidelity_selects_the_lod_tier(tmp_path: Path) -> None:
    """`--fidelity` dials the visual LOD tier URDF/SDF carry (RM-P0-FLEET-02/05)."""
    from astro_mine.fleet import importers
    from astro_mine.fleet._core import canonical_json

    urdf = tmp_path / "src.urdf"
    urdf.write_text(
        '<robot name="r"><link name="base">'
        '<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>'
        '<visual><geometry><sphere radius="0.5"/></geometry></visual></link></robot>',
        encoding="utf-8",
    )
    doc = importers.import_urdf(urdf, assets_dir=tmp_path / "a", uri_prefix="a/")
    sadf = tmp_path / "r.sadf.json"
    sadf.write_text(canonical_json(doc), encoding="utf-8")

    out = tmp_path / "coarse.urdf"
    assert run("export", str(sadf), "-o", str(out), "--fidelity", "massmodel") == 0
    assert ".lod2." in out.read_text(encoding="utf-8")


def test_export_load_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("export", str(tmp_path / "nope.yaml"), "-o", str(tmp_path / "o.urdf")) == 1
    assert "cannot read file" in capsys.readouterr().err


def test_export_reports_an_unexportable_asset(
    valid_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An asset the target format cannot express fails loudly, with the reason."""
    assert run("render", str(valid_file), "-o", str(tmp_path / "p.glb")) == 1
    assert "fleet render" in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["glb", "usd"])
def test_render_writes_a_preview(
    fmt: str, rover_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The acceptance criterion: `fleet render` produces a preview — no stub, no GPU, no network."""
    out = tmp_path / f"preview.{fmt}"
    assert run("render", str(rover_file), "-o", str(out), "--format", fmt) == 0
    assert out.is_file() and out.stat().st_size > 0
    assert "rendered astro-mine.fleet.prospecting-rover" in capsys.readouterr().out


def test_render_reports_the_proxy_substitution(
    rover_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mesh-free asset is previewed with inertia-equivalent boxes — never silently."""
    assert run("render", str(rover_file), "-o", str(tmp_path / "p.glb"), "--json") == 0
    rules = {loss["rule"] for loss in json.loads(capsys.readouterr().out)["losses"]}
    assert "render.proxy_geometry" in rules


def test_help_no_longer_calls_export_or_render_deferred(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The last acceptance criterion: the CLI must stop advertising these as deferred."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "export" in out and "render" in out
    assert "deferred" not in out.lower()
