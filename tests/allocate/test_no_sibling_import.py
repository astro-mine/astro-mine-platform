"""Narrow-waist contract test: the constraint builders import no sibling package (RM-P1-ALLOC-03).

"Builders consume the Link contact graph, Worlds traversability, Fleet budgets, and Prospect value
only via Core contracts / the injected ConstraintContext — no sibling package imports; Allocate
invents no physics" (allocate.md §6). This asserts the source of the whole ``constraints`` package
never imports ``astro_mine.{link,worlds,fleet,prospect,sim,...}`` — the inputs arrive only through
Core (``astro_mine.core``) contracts.
"""

from __future__ import annotations

import ast
import pathlib

import astro_mine.allocate.constraints as constraints_pkg

_SIBLING_PACKAGES = {
    "link",
    "worlds",
    "fleet",
    "prospect",
    "sim",
    "surrogate",
    "mind",
    "learn",
    "guard",
    "bench",
    "hub",
    "cloud",
}


def _imported_astro_mine_subpackages(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    seen: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("astro_mine.")
        ):
            seen.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("astro_mine."):
                    seen.add(alias.name.split(".")[1])
    return seen


def test_constraint_builders_import_no_sibling_package() -> None:
    pkg_dir = pathlib.Path(constraints_pkg.__file__).parent
    sources = sorted(pkg_dir.glob("*.py"))
    assert sources, "expected the constraints package to have source modules"
    for src in sources:
        subpackages = _imported_astro_mine_subpackages(src)
        offending = subpackages & _SIBLING_PACKAGES
        assert not offending, f"{src.name} imports sibling package(s) {sorted(offending)}"
        # Every astro_mine import is either Core or Allocate's own package.
        assert subpackages <= {"core", "allocate"}, f"{src.name} imports unexpected {subpackages}"
