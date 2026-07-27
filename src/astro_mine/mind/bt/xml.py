"""Groot-compatible behavior-tree XML — authoring/inspection format (RM-P1-MIND-02).

The concrete syntax for :mod:`astro_mine.mind.bt.model`: the BehaviorTree.CPP v4 XML dialect
(``<root BTCPP_format="4" main_tree_to_execute="...">`` wrapping a ``<BehaviorTree ID="...">``
with a single root node), authored/inspected in Groot and versioned with the stack spec
(mind.md §5). :func:`parse_behavior_tree` validates while it parses — unknown tags, missing
``ID``/``kind`` ports, empty composites, and bad enum values are rejected loudly (the
stack-spec loader's posture) — and :func:`to_xml` renders an AST back to canonical XML, so a
tree round-trips (``parse(to_xml(t)) == t``).

Portable node forms (a Groot-valid subset): composites ``<Sequence>`` / ``<Fallback>`` (and
the ``Reactive*`` aliases — the reactive executive re-ticks from the first child each tick
regardless); decorators ``<Inverter>`` / ``<ForceSuccess>`` / ``<ForceFailure>``; and the
generic leaves ``<Action ID=.. kind=planner|policy|primitive ..>`` and
``<Condition ID=.. kind=fresh_upstream>``. A planner/policy action names its tier via
``tier=..``; a primitive names its SADF action via ``action=..``.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from astro_mine.mind.bt.model import (
    ActionNode,
    BehaviorTree,
    BTNode,
    ConditionKind,
    ConditionNode,
    ControlKind,
    ControlNode,
    DecoratorKind,
    DecoratorNode,
    InvokeKind,
)

__all__ = ["BehaviorTreeXMLError", "parse_behavior_tree", "to_xml"]

_CONTROL_TAGS: dict[str, ControlKind] = {
    "Sequence": ControlKind.SEQUENCE,
    "ReactiveSequence": ControlKind.SEQUENCE,
    "Fallback": ControlKind.FALLBACK,
    "ReactiveFallback": ControlKind.FALLBACK,
}
_CONTROL_OUT: dict[ControlKind, str] = {
    ControlKind.SEQUENCE: "Sequence",
    ControlKind.FALLBACK: "Fallback",
}
_DECORATOR_TAGS: dict[str, DecoratorKind] = {
    "Inverter": DecoratorKind.INVERTER,
    "ForceSuccess": DecoratorKind.FORCE_SUCCESS,
    "ForceFailure": DecoratorKind.FORCE_FAILURE,
}
_DECORATOR_OUT: dict[DecoratorKind, str] = {v: k for k, v in _DECORATOR_TAGS.items()}
#: Attribute names with dedicated meaning, excluded from a leaf's free-form ``params``.
_RESERVED_ATTRS = frozenset({"ID", "kind", "tier", "action"})


class BehaviorTreeXMLError(Exception):
    """Raised when behavior-tree XML is malformed or fails structural validation."""


def parse_behavior_tree(source: str | bytes) -> BehaviorTree:
    """Parse and validate Groot BT XML into a :class:`BehaviorTree`.

    Raises :class:`BehaviorTreeXMLError` on any malformed or structurally invalid tree.
    """
    try:
        root = ET.fromstring(source.decode("utf-8") if isinstance(source, bytes) else source)
    except ET.ParseError as exc:
        raise BehaviorTreeXMLError(f"invalid BT XML: {exc}") from exc

    if root.tag == "root":
        fmt = root.get("BTCPP_format", "4")
        trees = root.findall("BehaviorTree")
        if not trees:
            raise BehaviorTreeXMLError("<root> contains no <BehaviorTree>")
        main = root.get("main_tree_to_execute")
        tree_el = _select_tree(trees, main)
    elif root.tag == "BehaviorTree":
        fmt = "4"
        tree_el = root
    else:
        raise BehaviorTreeXMLError(f"expected <root> or <BehaviorTree>, got <{root.tag}>")

    tree_id = tree_el.get("ID")
    if not tree_id:
        raise BehaviorTreeXMLError("<BehaviorTree> is missing its ID")
    children = list(tree_el)
    if len(children) != 1:
        raise BehaviorTreeXMLError(
            f"BehaviorTree {tree_id!r} must have exactly one root node, found {len(children)}"
        )
    return BehaviorTree(tree_id=tree_id, root=_parse_node(children[0]), format_version=fmt)


def _select_tree(trees: list[ET.Element], main: str | None) -> ET.Element:
    if main is None:
        if len(trees) != 1:
            raise BehaviorTreeXMLError(
                "multiple <BehaviorTree> present but no main_tree_to_execute set"
            )
        return trees[0]
    for tree in trees:
        if tree.get("ID") == main:
            return tree
    raise BehaviorTreeXMLError(f"main_tree_to_execute {main!r} matches no <BehaviorTree ID=..>")


def _parse_node(el: ET.Element) -> BTNode:
    tag = el.tag
    if tag in _CONTROL_TAGS:
        children = tuple(_parse_node(c) for c in el)
        if not children:
            raise BehaviorTreeXMLError(f"<{tag}> must have at least one child")
        return ControlNode(kind=_CONTROL_TAGS[tag], children=children, node_id=el.get("ID"))
    if tag in _DECORATOR_TAGS:
        kids = list(el)
        if len(kids) != 1:
            raise BehaviorTreeXMLError(f"<{tag}> decorator must have exactly one child")
        return DecoratorNode(
            kind=_DECORATOR_TAGS[tag], child=_parse_node(kids[0]), node_id=el.get("ID")
        )
    if tag == "Action":
        return _parse_action(el)
    if tag == "Condition":
        return _parse_condition(el)
    raise BehaviorTreeXMLError(f"unknown behavior-tree node <{tag}>")


def _parse_action(el: ET.Element) -> ActionNode:
    node_id = _require(el, "ID")
    kind = _require(el, "kind")
    try:
        invoke = InvokeKind(kind)
    except ValueError as exc:
        raise BehaviorTreeXMLError(f"Action {node_id!r}: unknown kind {kind!r}") from exc
    ref = _require(el, "action") if invoke is InvokeKind.PRIMITIVE else _require(el, "tier")
    return ActionNode(invoke=invoke, ref=ref, node_id=node_id, params=_params(el))


def _parse_condition(el: ET.Element) -> ConditionNode:
    node_id = _require(el, "ID")
    kind = _require(el, "kind")
    try:
        check = ConditionKind(kind)
    except ValueError as exc:
        raise BehaviorTreeXMLError(f"Condition {node_id!r}: unknown kind {kind!r}") from exc
    return ConditionNode(check=check, node_id=node_id, params=_params(el))


def _require(el: ET.Element, attr: str) -> str:
    value = el.get(attr)
    if not value:
        raise BehaviorTreeXMLError(f"<{el.tag}> is missing required attribute {attr!r}")
    return value


def _params(el: ET.Element) -> dict[str, str]:
    return {k: v for k, v in sorted(el.attrib.items()) if k not in _RESERVED_ATTRS}


def to_xml(tree: BehaviorTree) -> str:
    """Render ``tree`` back to canonical Groot BT XML (stable attribute order)."""
    root = ET.Element(
        "root", {"BTCPP_format": tree.format_version, "main_tree_to_execute": tree.tree_id}
    )
    bt = ET.SubElement(root, "BehaviorTree", {"ID": tree.tree_id})
    bt.append(_to_element(tree.root))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _to_element(node: BTNode) -> ET.Element:
    if isinstance(node, ControlNode):
        el = ET.Element(_CONTROL_OUT[node.kind], _id_attr(node.node_id))
        el.extend(_to_element(c) for c in node.children)
        return el
    if isinstance(node, DecoratorNode):
        el = ET.Element(_DECORATOR_OUT[node.kind], _id_attr(node.node_id))
        el.append(_to_element(node.child))
        return el
    if isinstance(node, ActionNode):
        attrs = {"ID": node.node_id, "kind": node.invoke.value}
        attrs["action" if node.invoke is InvokeKind.PRIMITIVE else "tier"] = node.ref
        attrs.update(node.params)
        return ET.Element("Action", attrs)
    attrs = {"ID": node.node_id, "kind": node.check.value, **dict(node.params)}
    return ET.Element("Condition", attrs)


def _id_attr(node_id: str | None) -> dict[str, str]:
    return {"ID": node_id} if node_id else {}
