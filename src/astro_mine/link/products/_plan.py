"""Assemble a Core ContactPlan from Link's contact windows + link budgets (RM-P0-LINK-04).

The reduction step from Link's own products to the **Core message catalog**: LINK-02's
:class:`~astro_mine.link.windows.ContactWindow` intervals become
:class:`~astro_mine.core.messages.ContactInterval`\\ s, annotated with LINK-03's
:class:`~astro_mine.link.budget.LinkBudget` (rate / latency / margin / mod-cod), over a
caller-declared contact graph of :class:`~astro_mine.core.messages.ContactNode`\\ s. Link
defines **no** new message types — it produces the Core ones Sim consumes, keeping to the
narrow waist (conventions.md §1.1). The assembled plan is checked against Core's
consumer-driven contract (:func:`~astro_mine.core.messages.validate_contact_plan`) so a
malformed product fails loudly at the boundary.

Backlog: RM-P0-LINK-04 -- astro-mine-link#4
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from astro_mine.core.messages import (
    ContactInterval,
    ContactNode,
    ContactPlan,
    validate_contact_plan,
)
from astro_mine.core.messages.model import LinkBudget as ContactLinkBudget
from astro_mine.core.units import EpochWindow
from astro_mine.link.budget import LinkBudget
from astro_mine.link.products._errors import LinkProductsError
from astro_mine.link.windows import ContactWindow

__all__ = ["build_contact_plan"]

#: The ordered node-pair a per-link budget is keyed by — matches ``ContactWindow``'s
#: ``(observer, target)``.
_Pair = tuple[str, str]


def build_contact_plan(
    nodes: Iterable[ContactNode],
    windows: Iterable[ContactWindow],
    *,
    budgets: Mapping[_Pair, LinkBudget] | None = None,
    epoch_window: EpochWindow | None = None,
    validate: bool = True,
) -> ContactPlan:
    """Assemble a content-addressed :class:`~astro_mine.core.messages.ContactPlan`.

    ``nodes`` declares the contact graph (id + space/ground role + free ``kind`` label —
    a role cannot be inferred from a name). ``windows`` are LINK-02 contact intervals;
    every window endpoint MUST be a declared node or :class:`LinkProductsError` is raised.
    ``budgets`` optionally maps an ordered ``(observer, target)`` pair to its LINK-03
    :class:`~astro_mine.link.budget.LinkBudget`, whose rate/latency/margin/mod-cod annotate
    that pair's intervals (a representative per-pass budget — the Phase-0 approximation; the
    full latency/bandwidth time-series is P1). ``epoch_window`` stamps the plan's validity
    span. With ``validate`` (the default) the result is checked against Core's
    :func:`~astro_mine.core.messages.validate_contact_plan` contract.

    Store-and-forward routes are out of scope (P1); the plan carries intervals + nodes only.
    """
    node_list = list(nodes)
    known = {node.id for node in node_list}
    budget_map = dict(budgets) if budgets is not None else {}

    intervals: list[ContactInterval] = []
    for window in windows:
        if window.observer not in known or window.target not in known:
            missing = {window.observer, window.target} - known
            raise LinkProductsError(
                f"contact window {window.observer!r}->{window.target!r} references "
                f"undeclared node(s) {sorted(missing)}; declare every node in the contact graph"
            )
        budget = budget_map.get((window.observer, window.target))
        intervals.append(_interval(window, budget))

    plan = ContactPlan(
        nodes=node_list,
        intervals=intervals,
        epoch_start_tdb_s=None if epoch_window is None else epoch_window.start.tdb_seconds,
        epoch_end_tdb_s=None if epoch_window is None else epoch_window.end.tdb_seconds,
        # Populate the additive typed wire field alongside the kept primitives (RFC-0007 Design §2:
        # producers SHOULD populate typed; consumers MUST prefer it).
        window=epoch_window,
    )
    if validate:
        validate_contact_plan(plan)
    return plan


def _interval(window: ContactWindow, budget: LinkBudget | None) -> ContactInterval:
    """One ``ContactInterval`` for ``window``, annotated with ``budget`` when present.

    The ``*_tdb_s`` primitives are kept and the additive typed ``window`` field is populated from
    the same :class:`~astro_mine.link.windows.ContactWindow` endpoints (RFC-0007 Design §2:
    producers SHOULD populate the typed field; consumers MUST prefer it)."""
    span = EpochWindow(start=window.start, end=window.end)
    if budget is None:
        return ContactInterval(
            node_a=window.observer,
            node_b=window.target,
            start_tdb_s=window.start.tdb_seconds,
            end_tdb_s=window.end.tdb_seconds,
            window=span,
        )
    return ContactInterval(
        node_a=window.observer,
        node_b=window.target,
        start_tdb_s=window.start.tdb_seconds,
        end_tdb_s=window.end.tdb_seconds,
        window=span,
        max_rate_bps=budget.rate_bps,
        min_latency_s=budget.latency_s,
        mean_latency_s=budget.latency_s,
        margin_db=budget.margin_db,
        modcod=budget.modcod,
        link_budget=_link_budget(budget),
    )


def _link_budget(budget: LinkBudget) -> ContactLinkBudget:
    """Project Link's :class:`LinkBudget` onto the Core message-catalog breakdown."""
    required_ebn0_db = (
        None
        if budget.ebn0_db is None or budget.margin_db is None
        else budget.ebn0_db - budget.margin_db
    )
    return ContactLinkBudget(
        eirp_dbw=budget.eirp_dbw,
        path_loss_db=budget.fspl_db,
        gt_db_per_k=budget.gt_db_per_k,
        required_ebn0_db=required_ebn0_db,
        margin_db=budget.margin_db,
        modcod=budget.modcod,
    )
