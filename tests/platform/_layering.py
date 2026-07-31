"""The component import graph, and the checks conventions.md §11 requires over it.

Consolidation removed the package boundaries that used to make a layering violation a packaging
error. What replaces them is this module: a static (AST, never import) survey of every
``astro_mine`` source file, plus one function per rule in §11. Each check returns the violations
it found rather than asserting, so :mod:`tests.platform.test_layering` can assert emptiness against
the real tree *and* prove — with a synthetic graph — that the check actually fires. A layering rule
whose test has never been seen to fail is a layering rule that is not being enforced.

Static analysis, deliberately: importing a component to inspect its dependencies is exactly the
thing under test, and a runtime survey cannot tell a module-scope import from a deferred one after
the fact.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "COMPANIONS",
    "COMPOSITION_ROOTS",
    "CORE",
    "CORE_DEPENDENCY_FLOOR",
    "Edge",
    "Import",
    "Survey",
    "check_companion_isolation",
    "check_core_isolation",
    "check_cycles",
    "check_forbidden_distributions",
    "check_private_imports",
    "check_recorded_edges",
    "check_surface_isolation",
    "components",
    "lateral_edges",
    "report",
    "survey",
]

#: Tier 0 — the waist. Depends on nothing (conventions.md §3.2).
CORE = "core"

#: Tier 1 — Core companions. Each may depend on Core and on nothing else.
COMPANIONS = frozenset({"spice", "seal"})

#: The third-party floor Core is allowed to stand on: schema and serialization runtimes only
#: (core.md §2 principle 3 — "no numpy-heavy, no torch, no sim engines"). ``google`` is protobuf's
#: import name and ``capnp`` is pycapnp's; ``referencing`` is jsonschema's resolver backend.
CORE_DEPENDENCY_FLOOR = frozenset(
    {"capnp", "google", "jsonschema", "pydantic", "referencing", "yaml"}
)

#: Distributions a component must never import. The CLI and the API *depend on* the platform, so an
#: import in this direction is a cycle across distributions; ``svcs`` is barred because a container
#: inside a component is a service locator (§3.3).
FORBIDDEN_DISTRIBUTIONS = ("astro_mine.cli", "astro_mine.api", "svcs")

#: The composition roots inside this distribution, and the only files in it that may import
#: ``svcs``. conventions.md §3.3 names four places the platform is assembled into an application;
#: two of them are other distributions (``astro-mine-cli``, ``astro-mine-api``) and these are the
#: two that live here.
#:
#: The allowlist is a literal path list rather than a pattern, and that is the point: adding a
#: composition root should require editing this line and defending it in review. A root is not a
#: component that happens to wire things — it is an *application entrypoint*, reached by ``python
#: -m`` or a container ``ENTRYPOINT`` and never imported by anything else in the tree.
#:
#: The CLI/API half of the rule has no exemption at all. A composition root may hold a container;
#: nothing here may import the distributions that depend on this one.
COMPOSITION_ROOTS = (
    "src/astro_mine/cloud/submission/harness.py",  # the Cloud in-pod worker
    "src/astro_mine/studio/orchestrate/worker/",  # Studio's orchestration worker
)

#: An import kind. ``runtime`` is module scope and costs import time (§8); ``type-only`` is under
#: ``TYPE_CHECKING``; ``deferred`` is inside a function body. All three are design dependencies and
#: all three count for acyclicity (§3.2 rule 4) — only ``runtime`` counts against import cost.
_RUNTIME = "runtime"
_TYPE_ONLY = "type-only"
_DEFERRED = "deferred"


@dataclass(frozen=True)
class Import:
    """One import statement, resolved to an absolute module path."""

    module: str
    names: tuple[str, ...]
    kind: str
    file: str
    line: int

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Edge:
    """An import that crosses from one component into another."""

    src: str
    dst: str
    module: str
    names: tuple[str, ...]
    kind: str
    file: str
    line: int

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"

    def __str__(self) -> str:
        return f"{self.location}: {self.src} -> {self.module}"


def _is_generated(path: Path) -> bool:
    """Whether ``path`` is machine-generated and therefore exempt from the import rules.

    ``protoc`` emits absolute imports of the private ``_proto`` package a ``.proto`` file's
    ``import`` line names, and the generated file is not ours to edit. The *specs* are reviewed;
    the emitted Python is a build artifact that happens to be committed.
    """
    return "_proto" in path.parts or path.name.endswith(("_pb2.py", "_pb2.pyi", "_pb2_grpc.py"))


def _module_name(path: Path, root: Path) -> str:
    """The dotted module path of ``path`` under the ``astro_mine`` package rooted at ``root``."""
    rel = path.relative_to(root.parent)
    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(module: str, *, is_package: bool) -> str:
    return module if is_package else module.rpartition(".")[0]


def _resolve(module: str | None, level: int, package: str) -> str | None:
    """Resolve a possibly-relative import to an absolute module path."""
    if not level:
        return module
    base = package.split(".")
    if level - 1 >= len(base):
        return None
    anchor = base[: len(base) - (level - 1)]
    return ".".join([*anchor, module]) if module else ".".join(anchor)


class _Walker(ast.NodeVisitor):
    """Collects imports, tagging each with the scope it was written in."""

    def __init__(self, package: str, file: str) -> None:
        self.package = package
        self.file = file
        self.imports: list[Import] = []
        self._type_checking = 0
        self._function = 0

    def _kind(self) -> str:
        if self._type_checking:
            return _TYPE_ONLY
        if self._function:
            return _DEFERRED
        return _RUNTIME

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        guarded = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not guarded:
            self.generic_visit(node)
            return
        self._type_checking += 1
        for child in node.body:
            self.visit(child)
        self._type_checking -= 1
        for child in node.orelse:
            self.visit(child)

    def _visit_function(self, node: ast.AST) -> None:
        self._function += 1
        self.generic_visit(node)
        self._function -= 1

    visit_FunctionDef = _visit_function  # type: ignore[assignment]
    visit_AsyncFunctionDef = _visit_function  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(Import(alias.name, (), self._kind(), self.file, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolve(node.module, node.level, self.package)
        if module is None:
            return
        names = tuple(alias.name for alias in node.names)
        self.imports.append(Import(module, names, self._kind(), self.file, node.lineno))


def components(root: Path) -> frozenset[str]:
    """Every component in the tree, derived from the package layout rather than a list.

    Reading the directory instead of hard-coding the seventeen names is what keeps this suite from
    quietly stopping short of a component someone adds later.
    """
    return frozenset(
        child.name
        for child in root.iterdir()
        if child.is_dir()
        and not child.name.startswith(("_", "."))
        and (child / "__init__.py").is_file()
    )


def _source_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or _is_generated(path):
            continue
        yield path


@dataclass(frozen=True)
class Survey:
    """One pass over the tree: the edges, every import, and each module's ``__all__``.

    Parsing is the expensive part on this working tree, so the three products are built together
    rather than by three walks that would each re-read a thousand files.
    """

    edges: list[Edge]
    imports: list[Import]
    exports: dict[str, frozenset[str]]


def _declared_exports(tree: ast.Module) -> frozenset[str] | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if isinstance(node.value, ast.List | ast.Tuple):
            return frozenset(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return None


def survey(root: Path) -> Survey:
    """Survey ``root`` (the ``astro_mine`` package directory) in a single pass."""
    known = components(root)
    edges: list[Edge] = []
    all_imports: list[Import] = []
    exports: dict[str, frozenset[str]] = {}
    for path in _source_files(root):
        src = path.relative_to(root).parts[0]
        module = _module_name(path, root)
        package = _package_of(module, is_package=path.name == "__init__.py")
        file = str(path.relative_to(root.parent.parent).as_posix())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        declared = _declared_exports(tree)
        if declared is not None:
            exports[module] = declared

        walker = _Walker(package, file)
        walker.visit(tree)
        for imp in walker.imports:
            all_imports.append(imp)
            parts = imp.module.split(".")
            if len(parts) < 2 or parts[0] != "astro_mine":
                continue
            dst = parts[1]
            if dst == src or dst not in known:
                continue
            edges.append(Edge(src, dst, imp.module, imp.names, imp.kind, file, imp.line))
    return Survey(edges, all_imports, exports)


def lateral_edges(edges: Iterable[Edge]) -> list[Edge]:
    """Tier-2 → tier-2 edges: the only ones §3.2 asks to be argued for.

    Core and the companions are excluded on both sides — any component may depend on all three
    without justification, and the companions' own constraints are a separate check.
    """
    return [
        edge
        for edge in edges
        if edge.src not in COMPANIONS
        and edge.dst not in COMPANIONS
        and edge.src != CORE
        and edge.dst != CORE
    ]


def check_cycles(edges: Iterable[Edge]) -> list[str]:
    """§3.2 rule 1 — the component import graph MUST be a DAG.

    Every kind of edge counts: an import deferred into a function body is still a design
    dependency (§3.2 rule 4).
    """
    graph: dict[str, set[str]] = {}
    for edge in edges:
        graph.setdefault(edge.src, set()).add(edge.dst)
        graph.setdefault(edge.dst, set())

    violations: list[str] = []
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    stack: list[str] = []

    def walk(node: str) -> None:
        colour[node] = GREY
        stack.append(node)
        for neighbour in sorted(graph[node]):
            if colour[neighbour] == GREY:
                cycle = stack[stack.index(neighbour) :]
                violations.append(" -> ".join([*cycle, neighbour]))
            elif colour[neighbour] == WHITE:
                walk(neighbour)
        stack.pop()
        colour[node] = BLACK

    for node in sorted(graph):
        if colour[node] == WHITE:
            walk(node)
    return sorted(set(violations))


def check_companion_isolation(edges: Iterable[Edge]) -> list[str]:
    """§3.2 — a companion MUST depend on Core only: not on another companion, not on a component.

    "A companion that needs a component is not a companion."
    """
    return [
        f"{edge.location}: companion {edge.src!r} imports {edge.module} "
        f"({'another companion' if edge.dst in COMPANIONS else 'a tier-2 component'})"
        for edge in edges
        if edge.src in COMPANIONS and edge.dst != CORE
    ]


def check_core_isolation(edges: Iterable[Edge], imports: Iterable[Import]) -> list[str]:
    """§3.2 — Core depends on nothing: no component, and no dependency above its declared floor."""
    violations = [
        f"{edge.location}: Core imports {edge.module} — the waist depends on nothing"
        for edge in edges
        if edge.src == CORE
    ]
    for imp in imports:
        if not imp.file.startswith("src/astro_mine/core/"):
            continue
        top = imp.module.split(".")[0]
        if top == "astro_mine" or top in sys.stdlib_module_names:
            continue
        if top not in CORE_DEPENDENCY_FLOOR:
            violations.append(
                f"{imp.location}: Core imports {top!r}, which is above its declared floor "
                f"({', '.join(sorted(CORE_DEPENDENCY_FLOOR))})"
            )
    return violations


def check_private_imports(
    edges: Iterable[Edge], exports: Mapping[str, frozenset[str]]
) -> list[str]:
    """§11 — no component may import another's underscore-private module, or a non-exported name.

    The two halves are one rule: a component's public surface is the modules it does not hide and
    the names it lists in ``__all__``. Reaching past either is reaching into an implementation.
    """
    violations: list[str] = []
    for edge in edges:
        tail = edge.module.split(".")[2:]
        if any(part.startswith("_") for part in tail):
            violations.append(
                f"{edge.location}: {edge.src!r} imports the private module {edge.module}"
            )
            continue
        declared = exports.get(edge.module)
        if declared is None:
            continue
        for name in edge.names:
            if name == "*" or name in declared:
                continue
            # A submodule of the target package is a module import, not a name lookup; the
            # private-module rule above already governs it.
            if f"{edge.module}.{name}" in exports or name.startswith("_"):
                if name.startswith("_"):
                    violations.append(
                        f"{edge.location}: {edge.src!r} imports the private name "
                        f"{edge.module}.{name}"
                    )
                continue
            violations.append(
                f"{edge.location}: {edge.src!r} imports {edge.module}.{name}, which is absent "
                f"from that module's __all__"
            )
    return violations


def check_forbidden_distributions(
    imports: Iterable[Import], roots: Sequence[str] = COMPOSITION_ROOTS
) -> list[str]:
    """§3.3 — no component imports the CLI or API distribution, and none imports ``svcs``.

    The ``svcs`` half is the one that matters most: it is what keeps the container at the
    composition roots and stops it becoming a service locator inside a component. ``roots`` are
    the files exempt from *that* half only — the CLI/API half admits no exemption, because an
    import of a distribution that depends on this one is a cycle wherever it appears.
    """
    violations: list[str] = []
    for imp in imports:
        at_root = any(imp.file.startswith(root) for root in roots)
        for forbidden in FORBIDDEN_DISTRIBUTIONS:
            if imp.module != forbidden and not imp.module.startswith(f"{forbidden}."):
                continue
            if forbidden == "svcs" and at_root:
                continue
            violations.append(
                f"{imp.location}: imports {imp.module} — a component MUST NOT import "
                f"{forbidden!r} (§3.3)"
            )
    return violations


def check_surface_isolation(repo_root: Path) -> list[str]:
    """§11 — a front-end surface package MUST NOT import another surface.

    A *surface* is an ``@astro-mine/<component>-ui`` package (§13). None ship from this
    distribution today — they live in ``astro-mine-console`` and the per-component UI repos — so
    this check finds nothing to inspect and says so by returning nothing. It is written against the
    layout rather than skipped, so the first surface added here is checked on arrival rather than
    on the day someone remembers the rule.
    """
    violations: list[str] = []
    for manifest in sorted(repo_root.glob("**/package.json")):
        if "node_modules" in manifest.parts:
            continue
        text = manifest.read_text(encoding="utf-8")
        if '"@astro-mine/' not in text or "-ui" not in text:
            continue
        package_dir = manifest.parent
        for source in sorted(package_dir.rglob("*.ts*")):
            if "node_modules" in source.parts:
                continue
            for line in source.read_text(encoding="utf-8").splitlines():
                if "@astro-mine/" in line and "-ui" in line and "import" in line:
                    violations.append(
                        f"{source.relative_to(repo_root)}: surface imports another surface "
                        f"— {line.strip()}"
                    )
    return violations


def check_recorded_edges(
    edges: Iterable[Edge], recorded: Mapping[tuple[str, str], str]
) -> list[str]:
    """§11 — a runtime lateral edge that is not recorded in the depending component's §6 fails.

    "The point is not to forbid lateral edges but to make adding one a deliberate, visible act."
    An unrecorded edge fails; a recorded one that has since disappeared also fails, because a table
    that outlives the thing it documents stops being read.
    """
    present = {
        (edge.src, edge.dst) for edge in lateral_edges(edges) if edge.kind == _RUNTIME
    }
    violations = [
        f"undeclared runtime lateral edge {src} -> {dst}: record it in {src}.md §6 with its "
        f"reason, or invert it (§3.2 rule 2)"
        for src, dst in sorted(present - set(recorded))
    ]
    violations += [
        f"{src} -> {dst} is recorded as a runtime lateral edge but no longer exists — drop it "
        f"from {src}.md §6 and from RECORDED_RUNTIME_EDGES"
        for src, dst in sorted(set(recorded) - present)
    ]
    return violations


def report(edges: Sequence[Edge]) -> str:
    """The runtime/type-only split §11 requires the suite to *report* without failing."""
    lines = ["", "component lateral edges (tier 2 -> tier 2)", "=" * 42]
    lateral = lateral_edges(edges)
    for kind in (_RUNTIME, _TYPE_ONLY, _DEFERRED):
        pairs: dict[tuple[str, str], int] = {}
        for edge in lateral:
            if edge.kind == kind:
                pairs[(edge.src, edge.dst)] = pairs.get((edge.src, edge.dst), 0) + 1
        total = sum(pairs.values())
        lines.append(f"\n{kind}: {total} import(s) across {len(pairs)} edge(s)")
        for (src, dst), count in sorted(pairs.items()):
            lines.append(f"    {src:>10} -> {dst:<10} {count:>3}")
    return "\n".join(lines)
