"""The Sim CLI (RM-P0-SIM-11, RM-P0-CLOUD-01).

Covers the two subcommands and the container back-compat shim: ``record`` round-trips a Sim
``Scenario`` file to MCAP deterministically; the legacy flat ``--scenario`` form (the Dockerfile +
Cloud entrypoint contract) still records; both the ``python -m astro_mine.sim`` and console-script
entry paths reach the same CLI; and ``run`` degrades with an actionable message — never a traceback
— at each missing layer (the ``[bench]`` loader, a content store).
"""

from __future__ import annotations

import re
import subprocess
import sys
import typing
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.sim.__main__ import main

DATA = Path(__file__).parent / "data"
_SCENARIO = DATA / "scenario.json"  # a materialized Sim Scenario (name/seed/dt_s/horizon/agents)
_ENV = "ASTRO_MINE_HUB_REGISTRY"
_TRACE_HASH = re.compile(r"[0-9a-f]{64}")  # Sim's Trace.content_hash — a bare sha256 hex digest


def _run_cli(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# --- record: the always-available local path ----------------------------------------------------


def test_record_roundtrips_a_sim_scenario(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "run.mcap"
    code, out, _ = _run_cli(
        "record", "--scenario-file", str(_SCENARIO), "--out", str(out_path), capsys=capsys
    )
    assert code == 0
    assert out_path.exists() and out_path.stat().st_size > 0
    assert _TRACE_HASH.fullmatch(out.strip())  # the determinism key


def test_record_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def _hash(name: str) -> str:
        code, out, _ = _run_cli(
            "record",
            "--scenario-file",
            str(_SCENARIO),
            "--seed",
            "7",
            "--out",
            str(tmp_path / name),
            capsys=capsys,
        )
        assert code == 0
        return out.strip()

    assert _hash("a.mcap") == _hash("b.mcap")  # same inputs + seed ⇒ same trace hash (CX-REPRO)


# --- container back-compat: the legacy flat form (Dockerfile:74 + Cloud) -------------------------


def test_legacy_scenario_flag_routes_to_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The container contract `--scenario X --seed N --out O` (no subcommand) still records."""
    out_path = tmp_path / "legacy.mcap"
    code, out, _ = _run_cli(
        "--scenario", str(_SCENARIO), "--seed", "7", "--out", str(out_path), capsys=capsys
    )
    assert code == 0
    assert out_path.exists()
    assert _TRACE_HASH.fullmatch(out.strip())


def test_top_level_help_is_not_shimmed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0  # -h/--help stay top-level, not routed to `record`


# --- both entry paths reach the same CLI --------------------------------------------------------


def test_console_script_is_registered() -> None:
    names = {ep.name for ep in entry_points(group="console_scripts")}
    assert "astro-mine-sim" in names


def test_python_m_entry_path_records(tmp_path: Path) -> None:
    """The container's `python -m astro_mine.sim` path works end-to-end (Dockerfile:74)."""
    out_path = tmp_path / "m.mcap"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "astro_mine.sim",
            "record",
            "--scenario-file",
            str(_SCENARIO),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    assert _TRACE_HASH.fullmatch(proc.stdout.strip())


# --- run: clean degradation, never a traceback (CX-LOCAL) ---------------------------------------


def test_run_without_bench_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate astro-mine-bench not being installed: `run` probes with find_spec, so make it report
    # the package as absent (matching a real uninstalled state, without touching the real env).
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "astro_mine.bench" else real(name, *a, **k),
    )
    code, out, err = _run_cli("run", "lunar-polar-ice-prospecting-v1", capsys=capsys)
    assert code == 2
    assert out == ""
    assert "astro-mine-sim[bench]" in err
    assert "Traceback" not in err


def test_run_without_a_registry_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    code, out, err = _run_cli("run", "lunar-polar-ice-prospecting-v1", capsys=capsys)
    assert code == 2
    assert out == ""
    assert _ENV in err or "--registry" in err
    assert "Traceback" not in err


def test_run_unknown_scenario_id_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = _run_cli(
        "run", "no-such-scenario", "--registry", str(tmp_path), capsys=capsys
    )
    assert code == 2
    assert "unknown scenario id" in err
    assert "Traceback" not in err


def test_run_without_hub_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import astro_mine.sim.runtime as runtime

    def _no_hub(*_a: object, **_k: object) -> object:
        raise ImportError("no hub client")

    monkeypatch.setattr(runtime, "open_bundle_store", _no_hub)
    code, out, err = _run_cli("run", "x", "--registry", str(tmp_path), capsys=capsys)
    assert code == 2
    assert out == ""
    assert "astro-mine-sim[hub]" in err
    assert "Traceback" not in err


def test_run_records_when_content_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The happy-path wiring: store → materialize → record → print the hash.

    The real content resolution is exercised by the Bench-adapter suite and manually against a local
    registry; here we mock the three boundaries to verify the CLI's orchestration end-to-end, since
    the anchor's published content is gated (#30/#56).
    """
    import astro_mine.sim.__main__ as cli
    import astro_mine.sim.bench as simbench
    import astro_mine.sim.runtime as runtime

    class _Run:
        scenario = object()
        world_provider = resource_field = connectivity = None
        content_hashes: typing.ClassVar[dict[str, str]] = {}
        # Fully resolved: nothing failed to rebuild, so the #67 diagnostic stays quiet.
        unresolved: typing.ClassVar[tuple[object, ...]] = ()

    class _Trace:
        content_hash = "a" * 64

    monkeypatch.setattr(runtime, "open_bundle_store", lambda *_a, **_k: object())
    monkeypatch.setattr(simbench, "materialize_bench_run", lambda _id, *, store, seed: _Run())
    monkeypatch.setattr(cli, "record_episode", lambda *_a, **_k: _Trace())
    code, out, _ = _run_cli(
        "run",
        "lunar-polar-ice-prospecting-v1",
        "--registry",
        str(tmp_path),
        "--out",
        str(tmp_path / "o.mcap"),
        capsys=capsys,
    )
    assert code == 0
    assert out.strip() == "a" * 64


def test_run_warns_when_a_pinned_provider_did_not_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blind run is reported at run start, not inferred later from an empty scorecard (#67).

    `run` warns and proceeds rather than refusing: this is the library tier, and recording a partial
    run is a legitimate ask. The *scoring* path refuses instead, because a scorecard is a claim.
    """
    import astro_mine.sim.__main__ as cli
    import astro_mine.sim.bench as simbench
    import astro_mine.sim.runtime as runtime
    from astro_mine.sim.runtime.content import UnresolvedProvider

    class _Run:
        scenario = object()
        world_provider = resource_field = connectivity = None
        content_hashes: typing.ClassVar[dict[str, str]] = {}
        unresolved = (
            UnresolvedProvider(
                content_id="shackleton-de-gerlache-v1",
                kind="world_provider",
                producer="astro-mine-worlds",
                consequence="no terrain, gravity or illumination",
            ),
        )

    class _Trace:
        content_hash = "b" * 64

    monkeypatch.setattr(runtime, "open_bundle_store", lambda *_a, **_k: object())
    monkeypatch.setattr(simbench, "materialize_bench_run", lambda _id, *, store, seed: _Run())
    monkeypatch.setattr(cli, "record_episode", lambda *_a, **_k: _Trace())

    code, out, err = _run_cli(
        "run",
        "lunar-polar-ice-prospecting-v1",
        "--registry",
        str(tmp_path),
        "--out",
        str(tmp_path / "o.mcap"),
        capsys=capsys,
    )

    assert code == 0  # a warning, not a refusal
    assert out.strip() == "b" * 64  # and the run still happened
    assert "shackleton-de-gerlache-v1" in err
    assert "astro-mine-worlds" in err  # the package that would fix it
    assert "Traceback" not in err  # actionable message, never a traceback (CX-LOCAL)


# --- SPICE kernels: the flag, the env, and the failure that used to be a traceback (#80) --------


def test_record_needs_no_kernels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The zero-prerequisite path stays zero-prerequisite: no metakernel, no complaint.

    This is the regression that matters most. `record` on a self-contained scenario is the one
    thing a reader can run with nothing downloaded and no account, and it must not acquire a NAIF
    kernel set as a prerequisite.
    """
    monkeypatch.delenv("ASTRO_MINE_SPICE_METAKERNEL", raising=False)
    out_path = tmp_path / "run.mcap"
    code, out, err = _run_cli(
        "record", "--scenario-file", str(_SCENARIO), "--out", str(out_path), capsys=capsys
    )
    assert code == 0
    assert _TRACE_HASH.fullmatch(out.strip())
    assert "metakernel" not in err.lower()


def test_record_furnishes_the_metakernel_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import astro_mine.spice as spice

    furnished: list[Path] = []
    monkeypatch.setattr(spice, "load_metakernel", lambda path, **_k: furnished.append(Path(path)))
    mk = tmp_path / "lunar.tm"
    mk.write_text("KPL/MK")

    code, _, _ = _run_cli(
        "record",
        "--scenario-file",
        str(_SCENARIO),
        "--out",
        str(tmp_path / "run.mcap"),
        "--metakernel",
        str(mk),
        capsys=capsys,
    )
    assert code == 0
    assert furnished == [mk]


def test_missing_metakernel_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run_cli(
        "record",
        "--scenario-file",
        str(_SCENARIO),
        "--out",
        str(tmp_path / "run.mcap"),
        "--metakernel",
        str(tmp_path / "absent.tm"),
        capsys=capsys,
    )
    assert code == 2
    assert "--metakernel" in err
    assert "ASTRO_MINE_SPICE_METAKERNEL" in err
    assert "naif.jpl.nasa.gov" in err
    assert "Traceback" not in err


def test_unfurnished_geometry_names_the_two_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this issue is about: geometry needed, no pool furnished.

    Previously a `SpiceGeometryError` traceback four frames deep inside the illumination model,
    which reads as "this is broken" rather than "you have not supplied kernels".
    """
    import astro_mine.sim.__main__ as cli
    import astro_mine.spice as spice

    monkeypatch.delenv("ASTRO_MINE_SPICE_METAKERNEL", raising=False)

    def _boom(*_a: object, **_k: object) -> None:
        raise spice.SpiceGeometryError("cannot resolve position of 'SUN' relative to 'MOON'")

    monkeypatch.setattr(cli, "record_episode", _boom)

    code, _, err = _run_cli(
        "record",
        "--scenario-file",
        str(_SCENARIO),
        "--out",
        str(tmp_path / "run.mcap"),
        capsys=capsys,
    )
    assert code == 2
    assert "cannot resolve position of 'SUN'" in err  # Spice's diagnosis is kept
    assert "--metakernel" in err  # and the remedy is added
    assert "ASTRO_MINE_SPICE_METAKERNEL" in err
    assert "Traceback" not in err


def test_metakernel_flag_is_on_both_verbs() -> None:
    """`run` and `record` share the flag, so the umbrella's surfaces cannot drift (RFC-0011 §3)."""
    import argparse

    from astro_mine.sim.__main__ import add_record_arguments, add_run_arguments

    for add in (add_run_arguments, add_record_arguments):
        parser = argparse.ArgumentParser()
        add(parser)
        assert "--metakernel" in parser.format_help()
