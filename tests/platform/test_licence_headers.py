# SPDX-License-Identifier: Apache-2.0
"""Every shipped source file carries an SPDX licence identifier (astro-mine-platform#35).

Apache-2.0 does **not** require this. The `LICENSE` file covers the repository and the grant is
complete without a line in each file. It is here because of what changes at the public flip: SBOM
and licence-scanning tooling — including the SBOMs this platform already generates through Seal —
infer per-file provenance from headers and report "unknown" without them, and a downstream that
vendors a single file has nothing travelling with it.

**A gate rather than a one-time sweep**, and expressed as a test rather than a workflow step. The
sweep is the easy half; an unenforced rule decays, and a file added tomorrow would quietly not carry
one. `tests/platform/` is where this repository already keeps that kind of check —
`test_typecheck_ratchet`, `test_layering`, `test_no_retired_names`, `test_config_blob_contract` —
so it runs under `scripts/test.py` and is verified on a workstation. That mattered here: the org's
Actions minutes are exhausted (astro-mine/.github#8), so a check living only in a workflow could not
have been observed working at all, and a check nobody has seen run is not a check.

**Scope is what ships.** The wheel is built by maturin with `python-source = "src"`, so `src/` is
exactly the distributed surface, and Guard's Rust core is compiled into it. `tests/` and `scripts/`
are deliberately out: they are not distributed, so neither the SBOM argument nor the vendoring
argument reaches them. The other three distributions (`astro-mine-cli`, `-api`, `-ui`) each need
their own sweep; this one cannot reach them.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
IDENTIFIER = "SPDX-License-Identifier: Apache-2.0"

#: Path segments marking a generated tree. Deliberately the same glob ruff's `extend-exclude`,
#: mypy's `astro_mine.*._proto.*` override and coverage's `omit` already treat as generated, rather
#: than a fourth opinion about the same files. A header in a file the generator rewrites is a header
#: that fights the generator.
#:
#: `_proto` only. astro-mine-platform#35 also names `src/generated/` and "the generated API client",
#: but neither is in this repository -- they are `astro-mine-ui`'s and `astro-mine-api`'s. Carrying
#: them here would have been an exclusion matching nothing, which
#: :func:`test_the_generated_exclusion_is_real` exists to refuse.
GENERATED = frozenset({"_proto"})


def _is_generated(path: pathlib.Path) -> bool:
    return bool(GENERATED & set(path.parts))


def _shipped_sources() -> list[pathlib.Path]:
    """Every non-generated source file that ends up in the distribution.

    Discovery is **what git tracks**, not what is on disk, and that distinction is load-bearing
    rather than fastidious. Walking `rust/` directly also descends into `rust/target/` -- Cargo's
    build directory, gitignored, holding prost output and the generated sources of third-party
    crates. This test passed in isolation and failed in the full suite for exactly that reason: the
    suite builds Guard's Rust core first, so the build directory exists by the time the gate runs.
    A gate whose answer depends on whether something was compiled first is not a gate.

    Asking git also settles the scope question for free. A tracked file is one this repository is
    responsible for; a build artifact is not, and neither is a dependency's generated code.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", "src/**/*.py", "rust/**/*.rs"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    found = [REPO / rel for rel in listed.stdout.split("\0") if rel]
    return sorted(p for p in found if not _is_generated(p))


def test_every_shipped_source_file_carries_the_identifier() -> None:
    """**The gate.** A new file without a header fails here, naming the file and the line to add."""
    sources = _shipped_sources()
    assert sources, "found no sources to check — the discovery above has broken, not the tree"

    missing = [
        p.relative_to(REPO).as_posix()
        for p in sources
        if IDENTIFIER not in p.read_text(encoding="utf-8", errors="replace")[:512]
    ]
    assert missing == [], (
        f"{len(missing)} shipped source file(s) carry no SPDX identifier: {missing[:10]}"
        f"{' …' if len(missing) > 10 else ''}. Add `{IDENTIFIER}` as the first line (after a "
        f"shebang, if there is one)."
    )


def test_the_identifier_precedes_the_module_docstring() -> None:
    """Position is load-bearing: a header placed wrong silently eats ``__doc__``.

    A module docstring has to be the *first statement* to become ``__doc__``. A comment above it is
    not a statement, so the header is invisible to Python — but the same header placed one line
    *below* the opening quotes would make the docstring an ordinary string expression and every
    ``help()``, Sphinx page and ``__doc__`` read would return ``None``. Checked rather than trusted,
    because the sweep that wrote 744 of these was mechanical and a mechanical error is uniform.
    """
    offenders: list[str] = []
    for path in (p for p in _shipped_sources() if p.suffix == ".py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines()[:4] if line.strip()]
        if not lines:
            continue
        header_at = next((i for i, line in enumerate(lines) if IDENTIFIER in line), None)
        doc_at = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith(('"""', "'''"))),
            None,
        )
        if header_at is not None and doc_at is not None and header_at > doc_at:
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], (
        f"the identifier sits inside or below the module docstring in: {offenders}. It must be a "
        f"comment *above* it, or the docstring stops being __doc__."
    )


@pytest.mark.parametrize("segment", sorted(GENERATED))
def test_the_generated_exclusion_is_real(segment: str) -> None:
    """An exclusion nothing matches is an exclusion that has silently stopped meaning anything.

    If a generated tree is renamed or removed, this fails rather than leaving a stale escape hatch
    that a future non-generated directory could wander into.
    """
    matches = [p for p in (REPO / "src").rglob("*.py") if segment in p.parts]
    assert matches, (
        f"no file under src/ lies in a {segment!r} directory, so excluding it excludes nothing. "
        f"If the generated tree moved, move this with it; if it is gone, drop the entry."
    )
