"""``astro-mine-mind`` CLI — validate / compose / stacks over the shipped reference stacks (G2.6).

The properties the issue is about:

* ``stacks`` enumerates the 6 reference stacks + 13 manifests **from package data** (wheel-safe);
* ``validate`` catches a stack that binds an **unregistered plugin**, naming the entry-point group
  and the missing name — the failure a shape-only validator misses;
* ``compose`` reports tier → plugin → **version**;
* there is **no ``run``** — stepping a stack needs a Core Environment Mind does not provide.
"""

from __future__ import annotations

import subprocess
import zipfile
from importlib import resources
from pathlib import Path

import pytest
import yaml

from astro_mine.mind import cli

ROOT = Path(__file__).resolve().parents[2]
_REFERENCE = "astro_mine.mind.reference"


# --------------------------------------------------------------------------- package data


def test_stacks_enumerates_package_data() -> None:
    stacks = list(cli.iter_stack_resources())
    manifests = list(cli.iter_manifest_resources())
    assert len(stacks) == 6, stacks
    assert len(manifests) == 13, manifests
    assert "lunar_prospecting.yaml" in stacks
    assert "lunar_prospecting_anchor.yaml" in stacks


def test_stacks_command_lists_counts(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stacks"]) == 0
    out = capsys.readouterr().out
    assert "reference stacks (6)" in out
    assert "reference manifests (13)" in out


# --------------------------------------------------------------------------- validate


def test_validate_reference_stack_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", "lunar_prospecting.yaml"]) == 0  # bare name → package data
    assert "OK" in capsys.readouterr().out


def test_validate_names_unregistered_plugin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = yaml.safe_load(
        resources.files(_REFERENCE).joinpath("stacks/lunar_prospecting.yaml").read_text()
    )
    doc["stack_spec"]["tiers"][0]["plugin"] = "no.such.plugin"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")

    assert cli.main(["validate", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "no.such.plugin" in err
    assert "astro_mine.mind.tier_plugins" in err  # the group is named


def test_validate_fails_on_bad_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("stack_spec_version: '0.1'\nstack_spec: {}\n", encoding="utf-8")
    assert cli.main(["validate", str(bad)]) == 1


def test_validate_multiple_files_exit_nonzero_if_any_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a stack\n", encoding="utf-8")
    assert cli.main(["validate", "lunar_prospecting.yaml", str(bad)]) == 1


# --------------------------------------------------------------------------- compose


def test_compose_reports_tier_plugin_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compose", "lunar_prospecting.yaml"]) == 0
    out = capsys.readouterr().out
    assert "mind.reference.mission @" in out
    assert "shield" in out
    assert "astro_mine.mind.tier_plugins" in out
    assert "core interface versions:" in out


def test_compose_behavior_tree_stack(capsys: pytest.CaptureFixture[str]) -> None:
    # A behavior-tree-execution stack composes by auto-resolving the packaged reference tree.
    assert cli.main(["compose", "lunar_prospecting_bt.yaml"]) == 0
    assert "execution: behavior_tree" in capsys.readouterr().out


# --------------------------------------------------------------------------- no run verb


def test_no_run_verb() -> None:
    # `run` is deliberately absent — a real episode needs a Core Environment (Sim). argparse exits 2
    # on an unknown subcommand.
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "lunar_prospecting.yaml"])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- the wheel boundary


def test_wheel_packages_cli_stacks_and_entry_point(tmp_path: Path) -> None:
    """The CLI, the reference stacks/manifests, and the console-script entry point ship in a wheel.

    ``stacks``/``compose`` resolve reference data with ``importlib.resources``; a consumer installs
    a wheel, not this checkout. Inspecting a real wheel is the only proof the data and the entry
    point are actually packaged (the #55 / astro-mine-bench#37 wheel trap).
    """
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not build a wheel: {exc}")

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as whl:
        names = set(whl.namelist())
        assert "astro_mine/mind/cli.py" in names
        stacks = [n for n in names if "/reference/stacks/" in n and n.endswith(".yaml")]
        manifests = [n for n in names if "/reference/manifests/" in n and n.endswith(".yaml")]
        assert len(stacks) == 6, stacks
        assert len(manifests) == 13, manifests
        entry_points = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
        registered = whl.read(entry_points).decode().replace(" ", "")
        assert "astro-mine-mind=astro_mine.mind.cli:main" in registered
