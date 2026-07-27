"""The claims that only a real installation can prove.

Everything else in this suite injects entry points, which is fast and precise but shares one
blind spot: it never exercises the packaging metadata that makes the whole mechanism work in the
first place. So this module does it the long way — build the wheel, create an empty virtualenv,
install the umbrella plus an unrelated third-party distribution, and drive the console script as
a user would.

Three claims live or die here:

* **No PR to extend** (RFC-0011 §3) — ``am-cli-test-provider`` is not an ``astro-mine-*`` package
  and this repo contains no reference to its verbs or its scaffold kinds, yet ``astro-mine demo``
  and ``astro-mine new demo-doc`` both work.
* **Listing imports nothing** (RFC-0011 §1a) — asserted in a clean interpreter, where a stray
  import cannot be masked by another test having already imported the module.
* **A scaffolded plugin is a real plugin** (RFC-0011 §7) — ``astro-mine plugin new cli`` writes a
  package that installs, registers, and is discovered through its entry-point group. Nothing short
  of installing it proves that, and it is the acceptance criterion the feature stands on.

Marked ``integration`` because it shells out and builds; it is not deselected in CI, where these
are the properties most worth defending.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVIDER = Path(__file__).resolve().parent / "fixtures" / "provider"

# The negative assertion, run inside the venv: build the top-level help — the operation most
# tempting to implement by loading every provider — and prove the provider stayed unimported.
_NO_IMPORT_PROBE = """
import sys
from astro_mine.cli import main
try:
    main(["--help"])
except SystemExit:
    pass
assert "am_cli_test_provider" not in sys.modules, "listing verbs imported a provider"
print("clean")
"""


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The umbrella, built the way a user would receive it. Built once for the whole module."""
    if shutil.which("uv") is None:  # pragma: no cover - environment-dependent
        pytest.skip("uv is not on PATH; these tests install packages the way CI and users do")
    dist = tmp_path_factory.mktemp("dist")
    _run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=REPO_ROOT)
    (built,) = dist.glob("*.whl")
    return built


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory, wheel: Path) -> Path:
    """A throwaway venv holding exactly the umbrella and the fixture provider."""
    venv = tmp_path_factory.mktemp("installed") / "venv"
    # The platform wheel is a binary (cp312, PyO3) build — pin the venv to the running
    # interpreter instead of the machine default (the original pure wheel ran anywhere).
    _run(["uv", "venv", "--python", sys.executable, str(venv)])
    _run(
        ["uv", "pip", "install", str(wheel), str(PROVIDER)],
        env={**os.environ, "VIRTUAL_ENV": str(venv)},
    )
    return venv


def test_a_third_party_distribution_contributes_a_working_verb(installed: Path) -> None:
    """The contract's whole point: a package this repo has never heard of adds a command."""
    result = _cli(installed, "demo", "hello", "--shout")
    assert result.returncode == 0
    assert result.stdout.strip() == "HELLO"


def test_the_verbs_exit_status_survives_the_process_boundary(installed: Path) -> None:
    assert _cli(installed, "demo", "x", "--exit-code", "7").returncode == 7


def test_the_passthrough_adapter_forwards_its_tail(installed: Path) -> None:
    """Style 2 — the cheap on-ramp for a component that already has `main(argv) -> int`. Pinned
    here because argparse.REMAINDER is quirky enough that component authors should copy something
    known to work rather than rediscover it."""
    result = _cli(installed, "passthrough", "score", "--flag", "sim")
    assert result.returncode == 0
    assert result.stdout.strip() == "component ran score with flag=sim"


def test_listing_verbs_imports_no_provider(installed: Path) -> None:
    """The laziness guarantee, in a clean interpreter. Nothing else would catch a regression:
    an eager `load()` produces correct output, just slower and with the whole platform imported."""
    result = _run([str(installed / "bin" / "python"), "-c", _NO_IMPORT_PROBE], check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("clean")


def test_help_lists_the_third_party_verb_with_its_provider(installed: Path) -> None:
    """A verb the manifest knows nothing about is still listed — described from its distribution
    metadata, which is free, rather than from its Subcommand, which would cost an import."""
    result = _cli(installed, "--help")
    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "am-cli-test-provider 0.1.0" in result.stdout


def test_a_malformed_provider_is_reported_as_a_packaging_bug(installed: Path) -> None:
    result = _cli(installed, "malformed")
    assert result.returncode == 2
    assert "does not satisfy the astro_mine.cli contract" in result.stderr
    assert "am-cli-test-provider" in result.stderr
    assert "Traceback" not in result.stderr


def test_python_m_is_equivalent_to_the_console_script(installed: Path) -> None:
    """Container entrypoints and `uv run` reach for the module form; it must not rot."""
    result = _run(
        [str(installed / "bin" / "python"), "-m", "astro_mine.cli", "demo", "hi"], check=False
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "hi"


def test_the_umbrella_still_pulls_in_nothing(installed: Path) -> None:
    """The platform wheel and the fixture provider resolve and co-install cleanly.

    The original astro-mine-cli assertion was an *exact* two-package set — the zero-dependency
    rule against a real resolver. The consolidated platform wheel carries the whole platform's
    dependency closure, so the exact-set claim is structurally per-repo; what survives is that
    the resolver accepts the pair and both distributions land."""
    uv = shutil.which("uv")
    assert uv is not None
    listing = _run(
        [uv, "pip", "list", "--format", "json"],
        env={**os.environ, "VIRTUAL_ENV": str(installed)},
    )
    names = {package["name"] for package in json.loads(listing.stdout)}
    assert {"astro-mine-platform", "am-cli-test-provider"} <= names


def test_a_third_party_distribution_contributes_a_scaffold_kind(
    installed: Path, tmp_path: Path
) -> None:
    """The no-PR-to-extend rule for scaffolds (RFC-0011 §7). `demo-doc` appears nowhere in this
    package's source — the umbrella learns of it from installed metadata and routes to its owner."""
    out = tmp_path / "authored.yaml"
    result = _cli(installed, "new", "demo-doc", str(out), "--marker", "proved")
    assert result.returncode == 0, result.stderr
    assert out.read_text(encoding="utf-8") == "kind: demo-doc\nmarker: proved\n"


def test_both_scaffold_groups_list_the_third_party_kinds(installed: Path) -> None:
    """Two groups, two listings, one metadata read each — a kind the manifest has never heard of
    is still shown, described from its distribution rather than by loading it."""
    documents = _cli(installed, "new")
    assert documents.returncode == 0
    assert "demo-doc" in documents.stdout
    assert "am-cli-test-provider 0.1.0" in documents.stdout

    plugins = _cli(installed, "plugin", "new")
    assert plugins.returncode == 0
    assert "demo-plugin" in plugins.stdout
    # The umbrella's own kind is listed alongside the third party's, from the same listing.
    assert "cli" in plugins.stdout


def test_a_scaffolded_verb_plugin_installs_registers_and_runs(
    installed: Path, wheel: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The acceptance criterion, end to end and in that order: scaffold, install, discover, run.

    A scaffold that emits something plausible but not installable is worse than none at all — the
    author debugs *our* template before writing a line of their own. So the generated package is
    put into a venv that holds nothing but the umbrella, and the verb it claims to register is
    invoked through the console script exactly as a user would.
    """
    workspace = tmp_path_factory.mktemp("scaffolded")
    package = workspace / "acme-greet"
    scaffolded = _cli(installed, "plugin", "new", "cli", str(package), "--verb", "greet")
    assert scaffolded.returncode == 0, scaffolded.stderr

    venv = workspace / "venv"
    # The platform wheel is a binary (cp312, PyO3) build — pin the venv to the running
    # interpreter instead of the machine default (the original pure wheel ran anywhere).
    _run(["uv", "venv", "--python", sys.executable, str(venv)])
    _run(
        ["uv", "pip", "install", str(wheel), str(package)],
        env={**os.environ, "VIRTUAL_ENV": str(venv)},
    )

    ran = _cli(venv, "greet", "--name", "moon")
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "hello, moon"
    # Discovered, not merely runnable: it shows up in the listing like any other verb.
    assert "greet" in _cli(venv, "--help").stdout


def _cli(venv: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run([str(venv / "bin" / "astro-mine"), *args], check=False)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:  # pragma: no cover - surfaced only on failure
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {' '.join(command)}")
    return result
