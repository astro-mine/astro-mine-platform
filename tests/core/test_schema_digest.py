"""An installed Core reports the schema digest the published bundle carries (#55).

``VERSIONING.md`` §4 freezes the Core interface version at ``0.1.0``, so version negotiation
is a no-op and reproducibility rests on *other* mechanisms — chief among them a
content-addressed digest of the schema set (§4.2; CX-REPRO). :data:`astro_mine.core.SCHEMA_DIGEST`
is that value, and Bench pins it (astro-mine-bench#39).

Two properties have to hold, and the second is the one with teeth:

1. The committed constant is **current** — it equals the digest of the schemas in the tree.
2. It survives **packaging**. The digest covers the ``.proto`` sources under ``schemas/proto/``,
   which live at the repo root and are *not* in the wheel. So the constant cannot be recomputed
   from an installed package, and any implementation that tried (e.g. a filesystem walk relative
   to ``__file__``) would look perfectly correct in this repo and ship a plausible-but-wrong
   digest to every consumer — who all install from a wheel built by ``uv`` from the git source.
   A test that only ever ran against the source checkout would pass while shipping that exact
   bug, so ``test_wheel_carries_the_digest`` builds a real wheel and reads the value back out.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from astro_mine.core import SCHEMA_DIGEST

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_schema_bundle.py"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_build_schema_bundle_digest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_builder()


def test_committed_digest_is_current(builder: ModuleType) -> None:
    """The constant matches the schemas actually in the tree.

    It is a derived value under version control — the same contract as ``uv.lock``. Editing a
    schema without regenerating it would have consumers pinning a digest no bundle carries."""
    computed = builder.schema_digest(builder.collect_schema_files())
    assert computed == SCHEMA_DIGEST, (
        f"astro_mine.core.SCHEMA_DIGEST is stale: {SCHEMA_DIGEST} != {computed}\n"
        "Regenerate it: uv run python scripts/build_schema_bundle.py --update-digest"
    )


def test_digest_equals_the_bundles(builder: ModuleType, tmp_path: Path) -> None:
    """The constant is the same value the published bundle advertises."""
    bundled = builder.build(tmp_path / "bundle")
    assert bundled == SCHEMA_DIGEST
    assert (tmp_path / "bundle" / "SCHEMA_DIGEST").read_text(encoding="utf-8").strip() == bundled


def test_digest_has_the_content_address_form() -> None:
    assert SCHEMA_DIGEST.startswith("sha256:")
    assert len(SCHEMA_DIGEST) == len("sha256:") + 64


def test_wheel_carries_the_digest(tmp_path: Path) -> None:
    """The digest survives into a **built wheel** — where the .proto sources do not.

    The load-bearing test. Every consumer gets Core as a wheel built by ``uv`` from the git
    source, and a fifth of the hashed inputs is absent from it. Reading the value back out of a
    real wheel is the only way to prove what consumers will actually see.
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
        assert "astro_mine/core/_schema_digest.py" in names, (
            "the generated digest module is missing from the wheel — an installed Core would "
            "not be able to report its schema digest at all"
        )
        # The .proto sources are deliberately absent; that absence is *why* the digest is a
        # committed constant. Assert it, so nobody 'fixes' this by walking the filesystem.
        # (Scoped to Core's own subtree: the platform wheel also carries Sim's
        # service proto, which Sim's wheel always shipped as package data.)
        core_protos = [
            n for n in names if n.endswith(".proto") and n.startswith("astro_mine/core/")
        ]
        assert not core_protos, (
            "a .proto source reached the wheel — if the packaged tree ever carries the full "
            "schema set, revisit whether the digest should be recomputed at runtime"
        )
        source = whl.read("astro_mine/core/_schema_digest.py").decode("utf-8")

    namespace: dict[str, object] = {}
    exec(compile(source, "_schema_digest.py", "exec"), namespace)
    assert namespace["SCHEMA_DIGEST"] == SCHEMA_DIGEST
