"""Hub's verbs on the umbrella CLI — `astro-mine publish|search|pull|verify`.

The artifact-registry actions RFC-0011 §2 puts at the top level: a contributor publishes, a
consumer searches and pulls, and anyone re-verifies — none of which requires knowing that Hub is
the package behind it (astro-mine/docs#57).

**Two Hub commands are deliberately not umbrella verbs.** ``resolve`` is a pin lookup that only
means anything in Hub's own vocabulary, and ``keygen`` mints key material; both read better as
`astro-mine-hub` commands than as platform verbs, and RFC-0011 §2's sketch lists neither. They
keep working exactly as before on the Hub CLI.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this
package and must not become one.

**Nothing here re-declares a flag.** Each adapter attaches the same ``add_*_arguments`` function
:mod:`astro_mine.hub.client.cli` uses, and calls the same handler — so a flag cannot appear on one
surface and not the other. That matters most for ``--trusted-key`` and ``--no-verify``: they decide
what a *verified* pull means, and a surface that silently lacked one would be a supply-chain
footgun rather than a missing convenience.
"""

from __future__ import annotations

import argparse

from astro_mine.hub.client.cli import (
    _cmd_publish,
    _cmd_pull,
    _cmd_search,
    _cmd_verify,
    add_publish_arguments,
    add_pull_arguments,
    add_search_arguments,
    add_verify_arguments,
)

__all__ = ["publish", "pull", "search", "verify"]


class _Publish:
    name = "publish"
    help = "publish a signed artifact to a registry"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_publish_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return int(_cmd_publish(args))


class _Search:
    name = "search"
    help = "discover artifacts in a registry"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_search_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return int(_cmd_search(args))


class _Pull:
    name = "pull"
    help = "pull and re-verify an artifact"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_pull_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return int(_cmd_pull(args))


class _Verify:
    name = "verify"
    help = "re-verify an artifact's supply chain"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_verify_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return int(_cmd_verify(args))


publish = _Publish()
search = _Search()
pull = _Pull()
verify = _Verify()
