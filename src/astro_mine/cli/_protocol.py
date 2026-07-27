"""The contribution contract — what an ``astro_mine.cli`` entry point must resolve to.

A component contributes a verb by pointing an entry point at an object with four members
(RFC-0011 §3). That is the whole contract::

    [project.entry-points."astro_mine.cli"]
    train = "astro_mine.learn.cli:umbrella"

**Why this is a structural Protocol and not a base class to inherit.** A component forced to
write ``from astro_mine.cli import Subcommand`` would take a runtime dependency on the umbrella —
inverting the layering this package exists to protect and making the umbrella a dependency of
every component (``conventions.md §1.1``). So conformance is checked by shape, at dispatch, and
the contract is a documented set of attribute names rather than an importable symbol. A component
may of course import this Protocol for type-checking only, under ``if TYPE_CHECKING``.

The cost of duck-typing is that a mistake surfaces late — which is why :func:`check_subcommand`
exists: a provider that does not conform is reported by **name, entry point and missing member**,
never as an ``AttributeError`` raised three frames into dispatch.

**One shape, three groups.** ``astro_mine.cli.scaffolds`` and ``astro_mine.cli.plugin_scaffolds``
(RFC-0011 §7; :mod:`astro_mine.cli._scaffolds`) bind to *this* contract rather than defining their
own: a scaffold is a thing with a name the user types, a one-line help, arguments of its own, and a
``run``. Inventing a second four-member protocol with the same members under different nouns would
have given component authors two shapes to learn and this package two checkers to keep in step. So
the checker is parameterized by the *group* and the *noun* it should blame — the message names the
group the author actually wrote into — and the shape stays one thing.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

__all__ = ["REQUIRED_MEMBERS", "InvalidSubcommandError", "Subcommand", "check_subcommand"]


@runtime_checkable
class Subcommand(Protocol):
    """What an ``astro_mine.cli`` entry point resolves to."""

    #: The verb as it appears on the command line. Must equal the entry point's name — the
    #: umbrella routes on the entry-point name, and a mismatch would make the object's own
    #: ``name`` a lie in help output and error messages.
    name: str

    #: One line, lower-case, no trailing period — it is rendered in a list.
    help: str

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this verb's arguments to a parser the umbrella owns.

        Called **only** when this verb is being run (or its ``--help`` requested), never to build
        the top-level listing.
        """

    def run(self, args: argparse.Namespace) -> int:
        """Execute the verb. The return value becomes the process exit status."""


#: Checked in order, so the error names the first thing wrong rather than the last.
REQUIRED_MEMBERS: tuple[str, ...] = ("name", "help", "add_arguments", "run")


class InvalidSubcommandError(Exception):
    """A provider resolved, but does not satisfy the contract.

    This is a packaging bug in the *providing* distribution, so the message names the
    distribution and the entry point — the two things needed to file the issue against the right
    repo — and never asks the user to read the umbrella's source.
    """


def check_subcommand(
    obj: Any,
    *,
    verb: str,
    entry: EntryPoint | None = None,
    contract: str = "astro_mine.cli",
    noun: str = "verb",
) -> Subcommand:
    """Return ``obj`` if it satisfies :class:`Subcommand`, else raise with a usable message.

    ``entry`` is optional so the checker is usable on an object that did not come from an entry
    point (the tests do this); when present, the error names the entry point's target and its
    distribution.

    ``contract`` and ``noun`` exist because the same four members are the contract for three
    entry-point groups. They only ever change the *message*, never what is checked: a scaffold that
    forgot ``run`` should be told it broke the ``astro_mine.cli.scaffolds`` contract, not sent
    looking for a verb it never wrote.
    """
    missing = [member for member in REQUIRED_MEMBERS if not hasattr(obj, member)]
    if missing:
        raise InvalidSubcommandError(
            f"the provider of {verb!r} does not satisfy the {contract} contract: "
            f"missing {', '.join(repr(m) for m in missing)}. {_blame(entry, noun)}"
        )
    for member in ("add_arguments", "run"):
        if not callable(getattr(obj, member)):
            raise InvalidSubcommandError(
                f"the provider of {verb!r} does not satisfy the {contract} contract: "
                f"{member!r} is not callable. {_blame(entry, noun)}"
            )
    return obj  # type: ignore[no-any-return]


def _blame(entry: EntryPoint | None, noun: str = "verb") -> str:
    """Point at whoever has to fix it — the providing distribution, not the user."""
    if entry is None:
        return f"A {noun} must provide: " + ", ".join(REQUIRED_MEMBERS) + "."
    dist = getattr(entry.dist, "name", None)
    origin = f"{entry.value!r} (from {dist!r})" if dist else repr(entry.value)
    return (
        f"Its entry point is {origin}; report this to that package. "
        f"A {noun} must provide: " + ", ".join(REQUIRED_MEMBERS) + "."
    )
