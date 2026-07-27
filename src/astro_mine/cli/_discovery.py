"""Verb discovery — read installed metadata, import nothing.

The umbrella depends on a **group name**, ``astro_mine.cli``, and never on a provider
(RFC-0011 §1a). Enumerating that group reads distribution metadata that ``pip``/``uv`` wrote at
install time; it does not execute a line of the providing package. That is what lets
``astro-mine --help`` stay cheap on a machine with the whole platform installed, and what lets a
third party add a verb without a PR here.

**Laziness is the feature, and it is easy to lose.** Anything in this module that touches
:meth:`~importlib.metadata.EntryPoint.load` outside :func:`load_verb` breaks the guarantee for
every caller, silently — the only symptom is a slower ``--help``. The tests assert the negative
(no provider in ``sys.modules`` after a help run) precisely because nothing else would catch it.

Modelled on :mod:`astro_mine.allocate.solvers.registry`, the platform's other entry-point
registry, including its stance on collisions: a name claimed twice is a hard error naming both
claimants, not a precedence rule the user has to learn.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import EntryPoint, entry_points

from astro_mine.cli._protocol import Subcommand, check_subcommand

__all__ = [
    "VERB_ENTRY_POINT_GROUP",
    "VerbCollisionError",
    "describe_provider",
    "discover_verbs",
    "load_verb",
]

#: The entry-point group a component advertises a verb under. The entry point's **name** is the
#: verb as typed; its value resolves to a :class:`~astro_mine.cli._protocol.Subcommand`.
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."astro_mine.cli"]
#:     train = "astro_mine.learn.cli:umbrella"
VERB_ENTRY_POINT_GROUP = "astro_mine.cli"


class VerbCollisionError(Exception):
    """Two installed distributions claim the same verb.

    Not resolved by precedence, on purpose. Which package handled a command is provenance: a
    silent winner would mean the same command line does different things on two machines with the
    same platform version, and the user would have no way to see it. Naming both claimants is the
    only outcome the user can act on.
    """


def discover_verbs(entries: Iterable[EntryPoint] | None = None) -> Mapping[str, EntryPoint]:
    """Every verb advertised in this environment, by name — **nothing is loaded**.

    ``entries`` is injectable so tests can build an environment without installing packages, and
    so the empty case is directly expressible. In production it is ``None`` and the group is read
    from installed metadata.

    Raises :class:`VerbCollisionError` if two distributions advertise the same verb.
    """
    found = entry_points(group=VERB_ENTRY_POINT_GROUP) if entries is None else tuple(entries)
    verbs: dict[str, EntryPoint] = {}
    for entry in found:
        clash = verbs.get(entry.name)
        if clash is not None:
            raise VerbCollisionError(
                f"the verb {entry.name!r} is claimed by both {describe_provider(clash)} and "
                f"{describe_provider(entry)}; uninstall one, or ask the packages to rename "
                f"their `{VERB_ENTRY_POINT_GROUP}` entry point"
            )
        verbs[entry.name] = entry
    return verbs


def describe_provider(entry: EntryPoint) -> str:
    """Name a verb's provider precisely enough to act on — distribution *and* target.

    Best-effort by construction: an entry point may advertise no resolvable distribution (a local
    editable install, a namespace package), and that is reported honestly rather than silently
    attributed to the umbrella itself.
    """
    dist = getattr(entry.dist, "name", None)
    version = getattr(entry.dist, "version", None)
    if dist and version:
        return f"{dist} {version} ({entry.value})"
    if dist:
        return f"{dist} ({entry.value})"
    return entry.value


def load_verb(entry: EntryPoint) -> Subcommand:
    """Import one provider and return its subcommand — **the only import in this package**.

    Called after the user has named a verb, so the import cost is paid for the command being run
    and nothing else. A provider whose own imports fail raises out of here unchanged: the umbrella
    must not swallow a component's ImportError into "unknown command", which would send the user
    hunting for a typo instead of the broken install they actually have.
    """
    return check_subcommand(entry.load(), verb=entry.name, entry=entry)
