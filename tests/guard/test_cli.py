"""``astro-mine-guard`` CLI + the packaged anchor spec (G2.6/G2.7, astro-mine-guard#29).

Two things are proven here:

* the reviewed anchor ``SafetySpec`` resolves **from package data**, via ``importlib.resources`` and
  never a path relative to the repo root — the #55 / astro-mine-bench#37 wheel trap, and the reason
  ``astro-mine-mind`` had to inline a second copy of a *safety* contract;
* the four verbs (``validate``/``compile``/``falsify``/``sign``) work, and fail **closed** — an
  invalid or unsigned-key path is a failure, never a pass.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from importlib import resources
from pathlib import Path

import pytest

from astro_mine.guard import cli
from astro_mine.guard.reference import (
    ANCHOR_SAFETY_SPEC_RESOURCE,
    anchor_safety_spec_text,
    load_anchor_safety_spec,
)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- the packaged spec


def test_anchor_spec_resolves_from_package_data() -> None:
    """The one copy of the anchor spec loads and validates from package data (wheel-safe)."""
    document = load_anchor_safety_spec()
    assert document.safety.id
    assert document.safety.constraints
    # Same bytes via the text accessor, resolved through importlib.resources.
    assert anchor_safety_spec_text() == (
        resources.files("astro_mine.guard.reference")
        .joinpath(ANCHOR_SAFETY_SPEC_RESOURCE)
        .read_text(encoding="utf-8")
    )


def test_no_source_resolves_the_anchor_from_a_checkout_path() -> None:
    """No shipped module reaches the anchor via ``examples/`` or a ``parents[N]`` walk to it.

    That path only exists in a git checkout; an installed wheel has no ``examples/`` sibling. Guard
    was bitten by exactly this — the spec lived outside ``src/`` and never reached the wheel.
    """
    src = ROOT / "src" / "astro_mine" / "guard"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # The executable anti-pattern is a path join to a sibling ``examples/`` dir — not a
        # docstring mentioning where the spec used to live.
        assert '/ "examples"' not in text, f"{path} resolves a path via a sibling examples/ dir"


# --------------------------------------------------------------------------- validate


def test_validate_anchor_ok() -> None:
    assert cli.main(["validate", "anchor"]) == 0


def test_validate_reports_actionable_error_and_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.safety.yaml"
    bad.write_text('safety_version: "0.1"\nsafety:\n  id: x\n', encoding="utf-8")
    assert cli.main(["validate", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err
    # names the missing fields (the JSON-Schema layer), not a bare traceback
    assert "required property" in err


def test_validate_defaults_to_the_anchor() -> None:
    assert cli.main(["validate"]) == 0  # no path → the shipped anchor


# --------------------------------------------------------------------------- compile


def test_compile_emits_artifact_and_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "anchor.compiled.json"
    assert cli.main(["compile", "anchor", "--out", str(out)]) == 0
    model = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(model, dict)
    err = capsys.readouterr().err
    assert "compiled_hash: sha256:" in err
    assert "spec_hash: sha256:".replace(" ", "") in err.replace(" ", "")


def test_compile_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    cli.main(["compile", "anchor", "--out", str(a)])
    cli.main(["compile", "anchor", "--out", str(b)])
    assert a.read_bytes() == b.read_bytes()  # content-addressed ⇒ byte-identical


# --------------------------------------------------------------------------- sign


def test_sign_validates_first_and_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_mine.guard.spec import generate_keypair

    priv, pub = generate_keypair()
    key = tmp_path / "k.key.pem"
    pubkey = tmp_path / "k.pub.pem"
    key.write_bytes(priv)
    pubkey.write_bytes(pub)

    code = cli.main(["sign", "anchor", "--key", str(key), "--pub", str(pubkey), "--verify"])
    assert code == 0
    out = capsys.readouterr().out
    assert "content_hash: sha256:" in out
    assert "verified:     True" in out


def test_sign_refuses_missing_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["sign", "anchor", "--key", str(tmp_path / "absent.pem")])
    assert code == 1
    assert "no signing key" in capsys.readouterr().err


def test_sign_refuses_invalid_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from astro_mine.guard.spec import generate_keypair

    priv, _pub = generate_keypair()
    key = tmp_path / "k.key.pem"
    key.write_bytes(priv)
    bad = tmp_path / "bad.safety.yaml"
    bad.write_text('safety_version: "0.1"\nsafety:\n  id: x\n', encoding="utf-8")
    assert cli.main(["sign", str(bad), "--key", str(key)]) == 1
    assert "refusing to sign an invalid spec" in capsys.readouterr().err


# --------------------------------------------------------------------------- falsify


def test_falsify_search_is_real_and_shield_holds(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip(
        "astro_mine.guard._core", reason="Rust safety core not built (maturin develop / uv sync)"
    )
    assert cli.main(["falsify", "--trials", "2", "--horizon", "40"]) == 0
    out = capsys.readouterr().out
    assert "the search is real" in out  # the unshielded control breached (non-vacuous)
    assert "shield held across 2 seed(s)" in out


def test_falsify_accepts_anchor_like_its_three_siblings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`falsify anchor` used to be `unrecognized arguments: anchor` (issue #35)."""
    pytest.importorskip("astro_mine.guard._core", reason="Rust safety core not built")
    assert cli.main(["falsify", "anchor", "--trials", "1", "--horizon", "40"]) == 0
    assert "shield held across 1 seed(s)" in capsys.readouterr().out


def test_falsify_runs_against_a_spec_the_user_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The step the authoring loop was missing: falsify what `astro-mine new safety` scaffolds.

    `validate → compile → falsify → sign` is what the guide teaches P2, and `falsify` was the one
    step that could not be performed on your own spec.
    """
    pytest.importorskip("astro_mine.guard._core", reason="Rust safety core not built")
    from astro_mine.guard.scaffolds import _safety_spec

    spec = tmp_path / "my.safety.yaml"
    spec.write_text(_safety_spec(spec_id="my-safety", name="My safety", scenario_ref=None))

    assert cli.main(["falsify", str(spec), "--trials", "2", "--horizon", "60"]) == 0
    captured = capsys.readouterr()
    assert "the search is real" in captured.out  # non-vacuous on a spec with no keep-out geometry
    assert "shield held across 2 seed(s)" in captured.out
    # The run names the spec it searched and the start it derived, so the result is attributable.
    assert "spec:    my-safety" in captured.err
    assert "inside their own bounds" in captured.err


def test_falsify_reports_a_bad_spec_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.safety.yaml"
    assert cli.main(["falsify", str(missing)]) == 1
    assert "cannot read" in capsys.readouterr().err

    bad = tmp_path / "bad.safety.yaml"
    bad.write_text('safety_version: "0.1"\nsafety:\n  id: x\n', encoding="utf-8")
    assert cli.main(["falsify", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "invalid spec" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------- the wheel boundary


def test_wheel_packages_the_anchor_spec_and_cli(tmp_path: Path) -> None:
    """The anchor spec, the CLI, and the console-script entry point survive into a built wheel.

    The bug this closes is a packaging bug: the spec used to live in ``examples/`` (a sibling of
    ``src/``) that maturin never packaged, so an installed Guard could not resolve it. Inspecting a
    real wheel is the only way to prove what a consumer actually gets — a test run in the checkout,
    where ``examples/`` still exists, would prove nothing.
    """
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not build a wheel: {exc}")

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as whl:
        names = set(whl.namelist())
        assert "astro_mine/guard/reference/safety_specs/anchor.safety.yaml" in names, (
            "the anchor SafetySpec is not package data — an installed Guard cannot resolve it"
        )
        assert "astro_mine/guard/cli.py" in names
        assert not any(n.startswith("examples/") for n in names), (
            "examples/ reached the wheel — the spec must live under the package, not beside it"
        )
        entry_points = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
        # maturin writes `name=target` without spaces; normalize before comparing.
        registered = whl.read(entry_points).decode().replace(" ", "")
        assert "astro-mine-guard=astro_mine.guard.cli:main" in registered
