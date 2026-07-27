"""``astro-mine-core validate`` — a checker for the Core-authored file formats.

Core owns nine hand-authored formats (SADF, ObjectiveSpec, MissionSpec, Plan/ContingentPlan,
plugin manifest, PolicyPackage, RunProvenance, the units vocabulary, the message catalog) and
until now shipped **no CLI** — a user authoring one of them had no way to ask "is this valid?"
short of writing Python. This module is the shell over the validators Core already ships
(``core.md §1``: *"types and validators"*), and it adds **no** new dependency: ``jsonschema``,
``pydantic``, ``pyyaml`` and ``referencing`` are all already Core dependencies.

**Dispatch is derived from the schema registry, never hand-maintained.** The set of known kinds,
their ``$id``s, and the schemas they validate against all come from
:data:`astro_mine.core.schemas.CORE_JSON_SCHEMAS` and :func:`astro_mine.core.schema_registry`
(RFC-0009). There is no ``{kind: filename}`` map here — that is exactly the fourth-inventory drift
the #50/#52/#53 cluster was about. Adding a tenth Core schema needs no change to this file; a test
(``tests/test_cli.py``) pins that.

**Honesty over a false pass.** A wrong-schema pass is worse than no checker, because it certifies a
document nobody validated. So the CLI never guesses a schema by *resemblance*: a document that
neither declares its ``$schema``, nor identifies itself completely (below), nor is given an
explicit ``--kind`` fails with the list of known kinds. And the two ``$defs``-only *vocabulary*
schemas (``units``, ``messages``) do not constrain a top-level document at all — validating a file
against them would pass anything — so they are not offered as document kinds; the authored message
documents (``action_batch``, ``contact_plan``) are.

**Two ways a document can be self-describing, because one is not always available.** A ``$schema``
pointer is the direct way. But a Core schema is ``additionalProperties: false`` at its root, so a
SADF document **cannot carry one** — the pointer that would identify it is the one key its own
format forbids. Requiring it would mean no SADF file on disk is routable by ``astro-mine validate``
(RFC-0011 §6), which is exactly the state this module was in. So a document is also accepted as
self-describing when it carries **every** root property its schema marks required **and** the
``<format>_version`` discriminator matches that schema's ``const`` exactly.

That is identification, not resemblance, and the difference is the whole of the rule. ``{"
objective_version": "0.1"}`` still fails — it *looks* like an objective and is not one, missing the
``objective`` member the format requires — and a test pins that it keeps failing. What passes is a
document that states which format and version it is, in that format's own required vocabulary, and
carries the structure to back the claim. Guard and Mind already route their formats this way
(``safety_version``, ``stack_spec_version``); this is the same rule, made stricter by also
demanding the rest of the required root.

The dispatch functions (:func:`iter_kinds`, :func:`resolve_kind`, :func:`validate_document`) are
importable so the umbrella ``astro-mine validate`` (RFC-0011) can route into them as a thin call
rather than a rewrite.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from functools import cache, lru_cache
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from astro_mine.core.schemas import CORE_JSON_SCHEMAS, core_schema, schema_registry

__all__ = [
    "Issue",
    "Kind",
    "iter_kinds",
    "main",
    "resolve_kind",
    "validate_document",
    "validate_source",
]

# JSON Schema keywords that constrain an *instance* (as opposed to ``$defs``/metadata, which only
# hold reusable subschemas). A schema declaring none of these — like the ``units`` and ``messages``
# vocabularies — accepts any document at the top level, so it is not a standalone document format.
_INSTANCE_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "$ref",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "enum",
        "const",
        "items",
        "prefixItems",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "if",
    }
)


@dataclasses.dataclass(frozen=True)
class Kind:
    """One validatable Core document format, derived from the schema registry.

    ``slug`` is the ``--kind`` name; ``schema_id`` is the canonical ``$id`` a document may declare
    in ``$schema``; ``schema`` is the resolved schema document; ``validator`` is the format's own
    ``validate_*`` loader (Pydantic + semantic checks), or ``None`` when only the JSON-Schema layer
    applies. ``constrains_instances`` is ``False`` for a ``$defs``-only vocabulary.
    """

    slug: str
    schema_id: str
    schema: dict[str, Any]
    package: str
    validator: Callable[[Any], None] | None

    @property
    def constrains_instances(self) -> bool:
        return bool(_INSTANCE_KEYWORDS & self.schema.keys())


@dataclasses.dataclass(frozen=True)
class Issue:
    """A single validation failure, in machine- and human-readable form."""

    layer: str  # "schema" (JSON Schema) or "model" (Pydantic/semantic)
    message: str
    pointer: str = ""  # RFC 6901 JSON Pointer to the offending location, "" for the root
    value: Any = None
    expected: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"layer": self.layer, "message": self.message}
        if self.pointer:
            out["pointer"] = self.pointer
        if self.expected:
            out["expected"] = self.expected
        return out

    def render(self) -> str:
        where = self.pointer or "<root>"
        line = f"  {where}: {self.message}"
        if self.expected:
            line += f" (expected {self.expected})"
        return line


def _loader_validators(package: str) -> dict[str, Callable[[Any], None]]:
    """The ``validate_*`` callables a format's ``<package>.loader`` defines, keyed by their slug.

    Discovered by introspection, not a table: the slug is the function name after ``validate_``.
    A package with no ``loader`` module (the ``units`` vocabulary) contributes nothing.
    """
    try:
        module = importlib.import_module(f"{package}.loader")
    except ModuleNotFoundError:
        return {}
    found: dict[str, Callable[[Any], None]] = {}
    for name in dir(module):
        if not name.startswith("validate_"):
            continue
        obj = getattr(module, name)
        # Only functions *defined in this loader* — never a re-exported symbol.
        if callable(obj) and getattr(obj, "__module__", None) == module.__name__:
            found[name[len("validate_") :]] = obj
    return found


@lru_cache(maxsize=1)
def _kinds() -> dict[str, Kind]:
    """Every validatable Core document format, derived from :data:`CORE_JSON_SCHEMAS`.

    One schema can yield several kinds (the message catalog yields ``action_batch`` and
    ``contact_plan``); a ``$defs``-only vocabulary with no document loader (``units``) yields none.
    """
    kinds: dict[str, Kind] = {}
    for package, filename in CORE_JSON_SCHEMAS:
        schema = core_schema(package, filename)
        schema_id = str(schema["$id"])
        validators = _loader_validators(package)
        if validators:
            for slug, fn in validators.items():
                kinds[slug] = Kind(slug, schema_id, schema, package, fn)
        elif _INSTANCE_KEYWORDS & schema.keys():
            # A document schema with no loader: JSON Schema is the whole gate. Slug from the stem.
            slug = filename.replace(".schema.json", "")
            kinds[slug] = Kind(slug, schema_id, schema, package, None)
        # else: a $defs-only vocabulary (units) — not a standalone document format.
    return dict(sorted(kinds.items()))


def iter_kinds() -> Iterator[Kind]:
    """Yield the known document kinds, slug-sorted. The single source of truth is the registry."""
    yield from _kinds().values()


def _schema_id_to_kinds() -> dict[str, list[Kind]]:
    index: dict[str, list[Kind]] = {}
    for kind in _kinds().values():
        index.setdefault(kind.schema_id, []).append(kind)
    return index


class KindError(ValueError):
    """Raised when the CLI cannot unambiguously determine which format a document is."""


def _known_kinds_hint() -> str:
    return ", ".join(k.slug for k in iter_kinds())


def _self_identifying_kinds(document: Any) -> list[Kind]:
    """The kinds a document identifies itself as, by declaring their required root completely.

    A schema qualifies as identifiable when it marks exactly one ``<format>_version`` property
    required with a ``const`` value — the discriminator every authored Core format carries. A
    document matches it only when **all** of that schema's required root properties are present and
    the discriminator equals the ``const``.

    Both halves matter. The version key alone would accept ``{"objective_version": "0.1"}`` — a
    fragment that resembles an objective without being one — and that is the resemblance-guessing
    this module refuses. Requiring the rest of the root is what turns the match into the document
    saying *"I am version 0.1 of this format"* in the format's own required vocabulary.

    Returns every match, so an ambiguous document is reported rather than resolved by order: the
    caller turns two matches into a ``--kind`` request, never a silent winner.
    """
    if not isinstance(document, dict):
        return []
    matches = []
    for kind in _kinds().values():
        required = kind.schema.get("required")
        properties = kind.schema.get("properties", {})
        if not required or not set(required) <= document.keys():
            continue
        discriminators = [
            name
            for name in required
            if name.endswith("_version")
            and isinstance(properties.get(name), dict)
            and "const" in properties[name]
        ]
        if len(discriminators) != 1:
            continue
        (version_key,) = discriminators
        if document.get(version_key) == properties[version_key]["const"]:
            matches.append(kind)
    return matches


def resolve_kind(document: Any, explicit: str | None) -> Kind:
    """Determine which :class:`Kind` a document is — from ``--kind`` or from the document itself.

    Precedence: an explicit ``--kind`` wins. Otherwise a ``$schema`` (or ``$id``) pointer is
    matched against the registry; failing that, the document is matched against the registry by its
    own declared format and version (:func:`_self_identifying_kinds`), which is the only route open
    to a format whose schema forbids a ``$schema`` key. A document that is neither self-describing
    nor given a kind, or whose pointer is unknown or ambiguous, raises :class:`KindError` naming the
    known kinds — it is **never** validated against a merely similar-looking schema.
    """
    kinds = _kinds()
    if explicit is not None:
        try:
            return kinds[explicit]
        except KeyError:
            raise KindError(
                f"unknown kind {explicit!r}; known kinds: {_known_kinds_hint()}"
            ) from None

    declared = None
    if isinstance(document, dict):
        raw = document.get("$schema") or document.get("$id")
        declared = raw if isinstance(raw, str) else None
    if declared is None:
        identified = _self_identifying_kinds(document)
        if len(identified) == 1:
            return identified[0]
        if len(identified) > 1:
            options = ", ".join(sorted(k.slug for k in identified))
            raise KindError(
                f"the document identifies itself as several formats ({options}); "
                f"pass --kind to say which"
            )
        raise KindError(
            "cannot determine the document kind: it declares no $schema, its root does not match "
            "any Core format's required members, and no --kind was given. "
            f"Pass --kind with one of: {_known_kinds_hint()}"
        )

    matches = _schema_id_to_kinds().get(declared)
    if not matches:
        raise KindError(
            f"$schema {declared!r} is not a known Core schema $id. "
            f"Pass --kind with one of: {_known_kinds_hint()}"
        )
    if len(matches) > 1:
        options = ", ".join(sorted(k.slug for k in matches))
        raise KindError(
            f"$schema {declared!r} names a catalog with several document kinds ({options}); "
            f"pass --kind to say which"
        )
    return matches[0]


def _pointer(error: ValidationError) -> str:
    return "".join(f"/{part}" for part in error.absolute_path)


def _expected(error: ValidationError) -> str:
    validator = error.validator
    value = error.validator_value
    if validator in {"type", "enum", "const"}:
        return f"{validator} {value!r}"
    if validator == "required":
        return f"required {value!r}"
    return str(validator)


@cache
def _validator_for(schema_id: str) -> Draft202012Validator:
    schema = next(k.schema for k in iter_kinds() if k.schema_id == schema_id)
    return Draft202012Validator(schema, registry=schema_registry(schema))


def validate_document(document: Any, kind: Kind) -> list[Issue]:
    """Validate an already-parsed document against ``kind``. Returns the issues, empty if valid.

    Two layers, most-actionable first:

    1. **JSON Schema** — structured errors (JSON-Pointer path, offending value, expectation),
       with cross-file ``$ref``s (``mission``/``messages`` → ``units``) resolved offline through
       :func:`schema_registry`. Only for a schema that constrains the instance; a structural
       failure short-circuits before the model layer.
    2. **The format's ``validate_*`` loader** — Pydantic and semantic checks that exceed JSON
       Schema (unique ids, unit strings, plugin-reference sanity). This is the authoritative layer
       for kinds whose top-level schema is a vocabulary (the message documents).

    A top-level ``$schema``/``$id`` is dispatch metadata, not document content — the Core schemas
    set ``additionalProperties: false`` and would reject it — so it is removed before validation.
    """
    if isinstance(document, dict) and document.keys() & {"$schema", "$id"}:
        document = {k: v for k, v in document.items() if k not in {"$schema", "$id"}}

    issues: list[Issue] = []
    if kind.constrains_instances:
        validator = _validator_for(kind.schema_id)
        for error in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path)):
            issues.append(
                Issue(
                    layer="schema",
                    message=error.message,
                    pointer=_pointer(error),
                    value=error.instance,
                    expected=_expected(error),
                )
            )
        if issues:
            return issues

    if kind.validator is not None:
        try:
            kind.validator(document)
        except Exception as exc:  # the loaders raise their own <X>ValidationError subclasses
            issues.append(Issue(layer="model", message=str(exc)))
    return issues


def validate_source(source: str | bytes, explicit_kind: str | None) -> tuple[Kind, list[Issue]]:
    """Parse a document (YAML is a JSON superset, so one parser does both) and validate it.

    Raises :class:`KindError` if the kind cannot be resolved and ``ValueError`` if the source is
    not a mapping. Otherwise returns the resolved kind and its issues (empty when valid).
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError(f"not parseable as YAML/JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("document must be a YAML/JSON mapping")
    kind = resolve_kind(document, explicit_kind)
    return kind, validate_document(document, kind)


# --------------------------------------------------------------------------- CLI


def _cmd_validate(args: argparse.Namespace) -> int:
    results: list[dict[str, Any]] = []
    failed = False
    for path in args.file:
        try:
            source = _read(path)
        except OSError as exc:
            failed = True
            _emit(args, path, None, [Issue("io", str(exc))], results)
            continue
        try:
            kind, issues = validate_source(source, args.kind)
        except (KindError, ValueError) as exc:
            failed = True
            _emit(args, path, None, [Issue("dispatch", str(exc))], results)
            continue
        if issues:
            failed = True
        _emit(args, path, kind, issues, results)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    return 1 if failed else 0


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _emit(
    args: argparse.Namespace,
    path: str,
    kind: Kind | None,
    issues: list[Issue],
    sink: list[dict[str, Any]],
) -> None:
    ok = not issues
    if args.json:
        sink.append(
            {
                "file": path,
                "kind": kind.slug if kind else None,
                "valid": ok,
                "issues": [issue.to_json() for issue in issues],
            }
        )
        return
    label = kind.slug if kind else "?"
    if ok:
        print(f"OK  {path}: valid {label}")
    else:
        print(f"FAIL {path}: {label}", file=sys.stderr)
        for issue in issues:
            print(issue.render(), file=sys.stderr)


def _cmd_kinds(args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps([{"kind": k.slug, "schema_id": k.schema_id} for k in iter_kinds()], indent=2)
        )
    else:
        width = max((len(k.slug) for k in iter_kinds()), default=0)
        for kind in iter_kinds():
            print(f"{kind.slug:<{width}}  {kind.schema_id}")
    return 0


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    """`validate`'s own arguments — attached to both this CLI and the umbrella's.

    Declared once so `astro-mine-core validate` and `astro-mine validate` (RFC-0011 §3, wired in
    :mod:`astro_mine.core.umbrella`) cannot drift apart. Note that ``--json`` is *not* here: it is
    a top-level flag on this parser, and the umbrella adapter re-adds it per verb, because on the
    umbrella surface a component has no top level to hang it from.
    """
    parser.add_argument("file", nargs="+", help="path to a JSON/YAML document ('-' for stdin)")
    parser.add_argument(
        "--kind",
        help="format to validate against (default: infer from the document's $schema). "
        "Run 'astro-mine-core kinds' for the list.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-mine-core",
        description="Validate Core-authored file formats (SADF, ObjectiveSpec, MissionSpec, "
        "Plan, plugin manifest, PolicyPackage, RunProvenance, message documents).",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate one or more documents")
    add_validate_arguments(validate)
    validate.set_defaults(func=_cmd_validate)

    kinds = sub.add_parser("kinds", help="list the known formats and their schema $ids")
    kinds.set_defaults(func=_cmd_kinds)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
