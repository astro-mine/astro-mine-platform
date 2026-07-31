"""The layering suite conventions.md §11 requires (RM-DIST — platform#5).

Components no longer sit behind package boundaries, so the rules of §1, §3.1, §3.2 and §3.3 are
only real if a test asserts them. "A layering rule that is only written down is a layering rule
that has already been broken somewhere."

Each rule appears twice here, and both halves matter:

* a **conformance** test, asserting the check finds nothing in this tree;
* a **firing** test, feeding the check a synthetic violation and asserting it complains.

Without the second, a check that silently stopped working would read as a passing suite forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.platform._layering import (
    COMPOSITION_ROOTS,
    CORE_DEPENDENCY_FLOOR,
    Edge,
    Import,
    Survey,
    check_companion_isolation,
    check_core_isolation,
    check_cycles,
    check_forbidden_distributions,
    check_private_imports,
    check_recorded_edges,
    check_surface_isolation,
    components,
    lateral_edges,
    report,
    survey,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "astro_mine"

#: Every runtime lateral edge (tier 2 → tier 2, module scope) this tree is allowed to have, with
#: the reason that justifies it — the in-repo mirror of each component document's §6. §11 fails the
#: build on an edge that is not here, and on an entry here that no longer describes a real edge.
#:
#: It is empty, and that is the outcome of platform#5 rather than an accident of a young codebase.
#: Three edges were listed here when this suite was written — ``allocate -> mind``, ``guard ->
#: mind``, ``sim -> bench`` — and each existed for the same reason: a shared abstraction living in
#: a component instead of at the waist. ``TierPlugin``, the guardrail report, the allocation
#: delegation contract and the episode-scoring vocabulary moved to Core, and all three dissolved
#: without a single behaviour change.
#:
#: Adding an entry is meant to be a deliberate act. It needs the matching §6 paragraph in the docs
#: repo, and a reviewer who agrees that §3.2 rule 2's test — "could the collaborator be a Protocol
#: the caller passes in?" — is genuinely answered *no*.
RECORDED_RUNTIME_EDGES: dict[tuple[str, str], str] = {}


@pytest.fixture(scope="module")
def tree() -> Survey:
    return survey(PACKAGE_ROOT)


# --- the tree conforms ---------------------------------------------------------------------


def test_component_graph_is_a_dag(tree: Survey) -> None:
    """§3.2 rule 1. A cycle is not a dependency to manage — it is two components that are one."""
    assert check_cycles(tree.edges) == []


def test_companions_depend_on_core_only(tree: Survey) -> None:
    """§3.2. Spice and Seal are companions precisely because they need nothing but the waist."""
    assert check_companion_isolation(tree.edges) == []


def test_core_depends_on_nothing(tree: Survey) -> None:
    """§3.2 + core.md §2.3. No component, and no dependency above the schema/serialization floor."""
    assert check_core_isolation(tree.edges, tree.imports) == []


def test_no_component_reaches_into_another_privately(tree: Survey) -> None:
    """§11. A component's public surface is its non-underscore modules and its ``__all__``."""
    assert check_private_imports(tree.edges, tree.exports) == []


def test_no_component_imports_the_cli_api_or_svcs(tree: Survey) -> None:
    """§3.3. The container lives at the composition roots; a component never sees one."""
    assert check_forbidden_distributions(tree.imports) == []


def test_only_the_composition_roots_import_svcs(tree: Survey) -> None:
    """§3.3. The exemption is real, and it is exactly two files wide.

    Asserted from the other direction as well, because an allowlist nobody checks the far side of
    grows: every ``svcs`` import in the tree must be at a listed root, and each listed root must
    still exist.
    """
    importers = {imp.file for imp in tree.imports if imp.module.split(".")[0] == "svcs"}
    assert importers, "the composition roots should be wiring with svcs; none imports it"
    for file in sorted(importers):
        assert any(file.startswith(root) for root in COMPOSITION_ROOTS), (
            f"{file} imports svcs but is not a declared composition root"
        )
    for root in COMPOSITION_ROOTS:
        assert (REPO_ROOT / root).exists(), f"declared composition root {root} no longer exists"


def test_no_surface_imports_another_surface() -> None:
    """§11 + §13. Vacuous here — no ``@astro-mine/<component>-ui`` package ships from this
    distribution — and live for the first one that does."""
    assert check_surface_isolation(REPO_ROOT) == []


def test_every_runtime_lateral_edge_is_recorded(tree: Survey) -> None:
    """§11. Adding a runtime lateral edge must be a deliberate, visible act."""
    assert check_recorded_edges(tree.edges, RECORDED_RUNTIME_EDGES) == []


def test_every_recorded_edge_carries_a_reason(tree: Survey) -> None:
    """"Convenient" is not a reason (§3.2 rule 2), and neither is an empty string."""
    for (src, dst), reason in RECORDED_RUNTIME_EDGES.items():
        assert len(reason) > 40, f"{src} -> {dst} is recorded without a real justification"


def test_the_scan_covers_every_component() -> None:
    """The component list is read from the tree, so a new component is checked on arrival."""
    found = components(PACKAGE_ROOT)
    assert "core" in found
    assert {"spice", "seal"} <= found
    assert len(found) >= 17


def test_lateral_edge_report(tree: Survey, capsys: pytest.CaptureFixture[str]) -> None:
    """§11 requires the suite to *report* runtime and type-only edges separately, without failing.

    The type-only and deferred counts are the interesting ones: a component with many of them and
    no runtime edges is telling you it wants its collaborators injected. Hub is the standing
    example — referenced by eight components, imported at runtime module scope by none.
    """
    with capsys.disabled():
        print(report(tree.edges))
    assert isinstance(lateral_edges(tree.edges), list)  # the non-failing half of §11


# --- every check fires ---------------------------------------------------------------------


def _edge(
    src: str, dst: str, module: str, *, kind: str = "runtime", names: tuple[str, ...] = ()
) -> Edge:
    return Edge(src, dst, module, names, kind, f"src/astro_mine/{src}/thing.py", 1)


def test_cycle_check_fires() -> None:
    cyclic = [_edge("sim", "bench", "astro_mine.bench"), _edge("bench", "sim", "astro_mine.sim")]
    assert check_cycles(cyclic)


def test_cycle_check_counts_deferred_imports() -> None:
    """§3.2 rule 4: deferring an import into a function body does not undo the dependency."""
    cyclic = [
        _edge("sim", "bench", "astro_mine.bench", kind="deferred"),
        _edge("bench", "sim", "astro_mine.sim", kind="type-only"),
    ]
    assert check_cycles(cyclic)


def test_companion_check_fires_on_component_import() -> None:
    assert check_companion_isolation([_edge("spice", "worlds", "astro_mine.worlds")])


def test_companion_check_fires_on_companion_import() -> None:
    assert check_companion_isolation([_edge("seal", "spice", "astro_mine.spice")])


def test_companion_check_allows_core() -> None:
    assert check_companion_isolation([_edge("spice", "core", "astro_mine.core.units")]) == []


def test_core_check_fires_on_component_import() -> None:
    assert check_core_isolation([_edge("core", "sim", "astro_mine.sim")], [])


def test_core_check_fires_above_the_floor() -> None:
    heavy = Import("numpy", ("array",), "runtime", "src/astro_mine/core/world/dem.py", 3)
    assert "numpy" not in CORE_DEPENDENCY_FLOOR
    assert check_core_isolation([], [heavy])


def test_core_check_allows_the_floor_and_the_stdlib() -> None:
    allowed = [
        Import("pydantic", ("BaseModel",), "runtime", "src/astro_mine/core/plan/model.py", 3),
        Import("dataclasses", ("dataclass",), "runtime", "src/astro_mine/core/plan/model.py", 4),
    ]
    assert check_core_isolation([], allowed) == []


def test_private_module_check_fires() -> None:
    edge = _edge("sim", "bench", "astro_mine.bench.metrics._trace", names=("EpisodeTrace",))
    assert check_private_imports([edge], {})


def test_non_exported_name_check_fires() -> None:
    edge = _edge("sim", "bench", "astro_mine.bench.metrics", names=("Undeclared",))
    assert check_private_imports([edge], {"astro_mine.bench.metrics": frozenset({"score"})})


def test_exported_name_passes() -> None:
    edge = _edge("sim", "bench", "astro_mine.bench.metrics", names=("score",))
    assert check_private_imports([edge], {"astro_mine.bench.metrics": frozenset({"score"})}) == []


@pytest.mark.parametrize(
    "module", ["astro_mine.cli", "astro_mine.cli.bench", "svcs", "svcs.Registry"]
)
def test_forbidden_distribution_check_fires(module: str) -> None:
    imp = Import(module, (), "runtime", "src/astro_mine/sim/runtime/episode.py", 7)
    assert check_forbidden_distributions([imp])


def test_a_composition_root_may_import_svcs() -> None:
    imp = Import("svcs", (), "runtime", "src/astro_mine/cloud/submission/harness.py", 44)
    assert check_forbidden_distributions([imp]) == []


def test_a_composition_root_may_not_import_the_cli() -> None:
    """The exemption covers the container, not the distribution boundary."""
    imp = Import("astro_mine.cli", (), "runtime", "src/astro_mine/cloud/submission/harness.py", 44)
    assert check_forbidden_distributions([imp])


def test_recorded_edge_check_fires_on_a_new_edge() -> None:
    assert check_recorded_edges([_edge("guard", "mind", "astro_mine.mind.registry")], {})


def test_recorded_edge_check_fires_on_a_stale_entry() -> None:
    assert check_recorded_edges([], {("guard", "mind"): "historical"})


def test_recorded_edge_check_ignores_non_runtime_edges() -> None:
    """Only module-scope edges are recorded: a deferred or type-only edge is the shape §3.2 rule 4
    calls a component asking for injection, not a dependency to argue for."""
    deferred = _edge("bench", "cloud", "astro_mine.cloud.sched", kind="deferred")
    assert check_recorded_edges([deferred], {}) == []
