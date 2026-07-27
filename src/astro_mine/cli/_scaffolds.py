"""Scaffold discovery — the federation behind `astro-mine new` and `plugin new` (RFC-0011 §7).

Scaffolding is a **cross-component authoring concern with no single-component home**: an asset is
Fleet's, a stack spec is Mind's, a SafetySpec is Guard's, and a solver plugin is Allocate's. So the
umbrella hosts the verbs and routes each kind to its owner — the same division `validate` already
makes, for the same reason (`conventions.md §1.1`: no component may import its siblings).

**The umbrella generates nothing it does not own.** It has no templating engine and no YAML parser,
and gaining either would break the zero-dependency rule that is this package's whole justification.
The owning component already has the schema, so it writes the bytes; this module only decides who
is asked. The one exception is the ``cli`` plugin kind (:mod:`astro_mine.cli._templates`) — the
umbrella owns the ``astro_mine.cli`` entry-point group, so it owns that group's scaffold, exactly
as it owns the ``validate`` verb by owning the routing problem.

**Routing is by name, not by inspection.** Unlike `validate` — which must ask each validator
*"is this file yours?"* because a path carries no owner — the user *types the kind*
(`astro-mine new asset`). So a scaffold is found by entry-point name, discovery stays pure metadata
reading, and `astro-mine new` can list every available kind without importing a single component.

**Two groups, not one with a flag.** Documents and plugins are separate groups rather than one
group whose members declare which verb they belong to, because reading such a declaration would
mean *loading every scaffold* to render `astro-mine new --help` — the precise cost
:mod:`astro_mine.cli._discovery` exists to avoid. A group name is free to filter on; an attribute
on a loaded object is not.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import EntryPoint, entry_points

from astro_mine.cli._discovery import describe_provider
from astro_mine.cli._protocol import Subcommand, check_subcommand

__all__ = [
    "DOCUMENT_SCAFFOLD_GROUP",
    "PLUGIN_SCAFFOLD_GROUP",
    "ScaffoldCollisionError",
    "discover_scaffolds",
    "load_scaffold",
]

#: Authored *documents* — the things a user writes and `astro-mine validate` later checks. The
#: entry point's **name** is the kind as typed (``asset``, ``stack``, ``safety``); its value
#: resolves to a :class:`~astro_mine.cli._protocol.Subcommand`.
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."astro_mine.cli.scaffolds"]
#:     asset = "astro_mine.fleet.umbrella:asset_scaffold"
DOCUMENT_SCAFFOLD_GROUP = "astro_mine.cli.scaffolds"

#: Installable *plugin packages* — a distribution that registers into one of the platform's live
#: extension groups. The entry point's **name** is the kind as typed (``solver``, ``tier``,
#: ``runner``), and names the plugin group being written *against*, not the group it is declared in.
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."astro_mine.cli.plugin_scaffolds"]
#:     solver = "astro_mine.allocate.umbrella:solver_scaffold"
PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


class ScaffoldCollisionError(Exception):
    """Two installed distributions offer the same scaffold kind.

    Held to the same stance as a verb collision (:class:`~astro_mine.cli.VerbCollisionError`) and
    for the same reason: which package generated a user's starting file is provenance. A silent
    winner would mean `astro-mine new asset` writes different bytes on two machines with nothing
    to tell them apart — and the divergence would be baked into whatever the user built on top.
    """


def discover_scaffolds(
    group: str, entries: Iterable[EntryPoint] | None = None
) -> Mapping[str, EntryPoint]:
    """Every scaffold kind advertised in ``group``, by name — **nothing is loaded**.

    ``entries`` is injectable so tests can build an environment without installing packages. In
    production it is ``None`` and the group is read from installed metadata, which costs no import.

    Raises :class:`ScaffoldCollisionError` if two distributions advertise the same kind.
    """
    found = entry_points(group=group) if entries is None else tuple(entries)
    kinds: dict[str, EntryPoint] = {}
    for entry in found:
        clash = kinds.get(entry.name)
        if clash is not None:
            raise ScaffoldCollisionError(
                f"the scaffold kind {entry.name!r} is offered by both {describe_provider(clash)} "
                f"and {describe_provider(entry)}; uninstall one, or ask the packages to rename "
                f"their `{group}` entry point"
            )
        kinds[entry.name] = entry
    return kinds


def load_scaffold(entry: EntryPoint, *, group: str) -> Subcommand:
    """Import one owner and return its scaffold — paid for by the command the user typed.

    A failure *inside* the owner's import propagates unchanged, as it does for a verb: turning a
    component's broken install into "unknown kind" would send the user hunting for a typo.
    """
    return check_subcommand(
        entry.load(), verb=entry.name, entry=entry, contract=group, noun="scaffold"
    )
