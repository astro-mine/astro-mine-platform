"""Validator discovery — the federation behind `astro-mine validate` (RFC-0011 §6).

Nine of the platform's authored formats belong to Core, but not all of them do: a SafetySpec is
Guard's, a stack spec is Mind's, and a WorldSpec will be Worlds'. RFC-0011 §6 settles the rule —
**the format's owner owns its validator; the umbrella federates them; no component reimplements
another's checker** — and this module is the federating half.

**The umbrella never parses a document.** It has no YAML parser and no schema library, and gaining
one would break the zero-dependency rule that is this package's whole justification. So routing is
inverted: instead of the umbrella reading a file to decide who owns it, it *asks each validator*
whether the file is theirs. Every owner already has the parser and the schema; none of that
knowledge has to be duplicated here, and adding a tenth format needs no change to this package.

**Every installed validator is imported when `validate` runs.** Ownership cannot be determined
without asking, and asking is a method call. That is a real cost and it is the honest one: it is
paid only by the command the user actually typed, never by `--help` or by any other verb. A
first-claim-wins shortcut would avoid some imports at the price of a silent precedence rule, which
is exactly the trade this platform refuses elsewhere (see the verb-collision stance in
:mod:`astro_mine.cli._discovery`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "VALIDATOR_ENTRY_POINT_GROUP",
    "ClaimCollisionError",
    "InvalidValidatorError",
    "Validator",
    "claim",
    "discover_validators",
]

#: The entry-point group a component advertises a validator under. The entry point's **name** is
#: the owning package's short name (``core``, ``guard``, ``mind``), used only in messages; the
#: routing is decided by :meth:`Validator.claims`, not by the name.
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."astro_mine.cli.validators"]
#:     guard = "astro_mine.guard.umbrella:validator"
VALIDATOR_ENTRY_POINT_GROUP = "astro_mine.cli.validators"


@runtime_checkable
class Validator(Protocol):
    """What an ``astro_mine.cli.validators`` entry point resolves to.

    Structural, for the same reason :class:`~astro_mine.cli.Subcommand` is: a component that had
    to import this Protocol would take a runtime dependency on the umbrella and invert the
    layering (``conventions.md §1.1``).
    """

    #: The owning package, for error messages (``"core"``, ``"guard"``, ``"mind"``).
    name: str

    def claims(self, path: str) -> str | None:
        """Return the format's name if this validator owns ``path``, else ``None``.

        Cheap and total: it is called for every installed validator on every file, so it must not
        raise on a document it does not recognize — a malformed file is the *validator's* error to
        report once it has claimed it, not a reason for routing to explode.
        """

    def validate(self, paths: Sequence[str], *, as_json: bool) -> int:
        """Validate the paths this validator claimed. Returns a process exit status."""


REQUIRED_MEMBERS = ("name", "claims", "validate")


class InvalidValidatorError(Exception):
    """A validator resolved but does not satisfy the contract — a packaging bug in its owner."""


class ClaimCollisionError(Exception):
    """Two validators claim the same document.

    Never resolved by precedence. Which checker judged a document is provenance: a silent winner
    would mean the same file validates differently depending on what else is installed, and
    nothing would tell the user which checker spoke.
    """


def discover_validators(entries: Iterable[EntryPoint] | None = None) -> tuple[Validator, ...]:
    """Load every advertised validator. ``entries`` is injectable for tests."""
    found = entry_points(group=VALIDATOR_ENTRY_POINT_GROUP) if entries is None else tuple(entries)
    return tuple(_check(entry.load(), entry) for entry in sorted(found, key=lambda e: e.name))


def claim(validators: Sequence[Validator], path: str) -> tuple[Validator, str]:
    """Find the one validator that owns ``path``.

    Raises :class:`ClaimCollisionError` if two claim it, and :class:`LookupError` if none does —
    the caller turns that into a message listing who *is* installed, which is more useful than a
    schema error from a checker that was never meant to read the file.
    """
    claims = [(v, kind) for v in validators if (kind := v.claims(path)) is not None]
    if len(claims) > 1:
        owners = ", ".join(f"{v.name} (as {kind})" for v, kind in claims)
        raise ClaimCollisionError(f"{path} is claimed by more than one validator: {owners}")
    if not claims:
        raise LookupError(path)
    return claims[0]


def _check(obj: Any, entry: EntryPoint) -> Validator:
    missing = [m for m in REQUIRED_MEMBERS if not hasattr(obj, m)]
    if missing:
        dist = getattr(entry.dist, "name", None)
        origin = f"{entry.value!r} (from {dist!r})" if dist else repr(entry.value)
        raise InvalidValidatorError(
            f"the validator {entry.name!r} does not satisfy the "
            f"{VALIDATOR_ENTRY_POINT_GROUP} contract: missing "
            f"{', '.join(repr(m) for m in missing)}. Its entry point is {origin}; report this to "
            f"that package. A validator must provide: {', '.join(REQUIRED_MEMBERS)}."
        )
    return obj  # type: ignore[no-any-return]
