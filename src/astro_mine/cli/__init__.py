"""Astro-Mine-CLI — the discoverable umbrella CLI.

One command, ``astro-mine <verb>``, in front of a platform that ships a CLI per component. A
user who does not yet know which package owns an action can guess the *action* and find it:
``astro-mine score``, ``astro-mine fetch``, ``astro-mine train``. Every component CLI keeps
working when invoked directly — the umbrella is the discoverable entry, **not** a replacement
(`RFC-0011 <https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md>`_;
``conventions.md §13``, normative).

**What this package is allowed to be.** A near-zero-dependency dispatcher: argparse and the
stdlib, and no runtime dependency on any Astro-Mine package — not even Core. RFC-0011 rejected
the umbrella-that-depends-on-everything because installing it to get one verb would drag the
whole platform onto the machine and break the local tier that must always work
(``conventions.md §7``).

**How a verb gets here.** A component declares an entry point in its *own* ``pyproject.toml``::

    [project.entry-points."astro_mine.cli"]
    train = "astro_mine.learn.cli:umbrella"

pointing at an object satisfying :class:`~astro_mine.cli.Subcommand` (``name``, ``help``,
``add_arguments``, ``run``). No component imports the umbrella, and the umbrella imports no
component to *list* it — discovery reads installed metadata, and a provider is imported only when
its verb actually runs. A third party gains an ``astro-mine <verb>`` the same way, with no change
to this package.

**Two more groups, the same contract.** ``astro_mine.cli.validators`` federates `astro-mine
validate` (RFC-0011 §6), and ``astro_mine.cli.scaffolds`` / ``astro_mine.cli.plugin_scaffolds``
federate `astro-mine new` and `astro-mine plugin new` (§7). All three verbs are *routers* — the
only kind of verb that can live here, since deciding which component owns a document or a kind is
the one job no component can do without importing its siblings.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from astro_mine.cli._discovery import (
    VERB_ENTRY_POINT_GROUP,
    VerbCollisionError,
    discover_verbs,
)
from astro_mine.cli._dispatch import build_parser, main
from astro_mine.cli._protocol import InvalidSubcommandError, Subcommand
from astro_mine.cli._scaffolds import (
    DOCUMENT_SCAFFOLD_GROUP,
    PLUGIN_SCAFFOLD_GROUP,
    ScaffoldCollisionError,
    discover_scaffolds,
)

__all__ = [
    "DOCUMENT_SCAFFOLD_GROUP",
    "PLUGIN_SCAFFOLD_GROUP",
    "VERB_ENTRY_POINT_GROUP",
    "InvalidSubcommandError",
    "ScaffoldCollisionError",
    "Subcommand",
    "VerbCollisionError",
    "__version__",
    "build_parser",
    "discover_scaffolds",
    "discover_verbs",
    "main",
]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source checkout without an install
    __version__ = "0.0.0.dev0"
