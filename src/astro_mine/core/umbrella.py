"""Core's validator for the umbrella CLI — the owner half of `astro-mine validate`.

Validation is a **headline** action: a user with a file in hand asks *"is this valid?"* without
knowing, or caring, which package owns the schema. RFC-0011 §6 settles who answers — **the
format's owner owns its validator; the umbrella federates them** — so Core no longer registers the
`validate` *verb*. It could never have federated: extending it would have meant Core consulting
Guard and Mind, i.e. Core importing its own consumers. The verb is now the umbrella's, and Core
contributes what it actually owns: `$id`-keyed dispatch over the nine Core-authored formats
(astro-mine/docs#57).

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). Core in particular must never depend on the
umbrella: it is the narrow waist, and the dependency would point the wrong way through the whole
platform.

**Claiming is cheap and total.** ``claims`` answers "is this mine?" for every file the umbrella
routes, so it resolves the kind and says no rather than raising: a document Core does not
recognize is Guard's or Mind's, not an error. A document that *is* Core's but malformed is claimed
and then reported properly by the real checker — the same one ``astro-mine-core validate`` runs,
which is what keeps the two surfaces from disagreeing about what is valid.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from astro_mine.core.cli import KindError, _cmd_validate, resolve_kind

__all__ = ["validator"]


class _CoreValidator:
    """Core's half of the federated `astro-mine validate`."""

    name = "core"

    def claims(self, path: str) -> str | None:
        """Resolve the document's kind from its ``$schema``/``$id``, or decline it.

        Unreadable or unparseable files are declined rather than claimed: at claim time nobody has
        agreed to own the file yet, so raising here would turn *another* component's malformed
        document into a Core traceback.
        """
        import yaml

        try:
            with open(path, encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            return None
        if not isinstance(document, dict):
            return None
        try:
            return str(resolve_kind(document, None).slug)
        except KindError:
            return None

    def validate(self, paths: Sequence[str], *, as_json: bool) -> int:
        """Run the same checker `astro-mine-core validate` runs — not a second implementation."""
        return int(_cmd_validate(argparse.Namespace(file=list(paths), kind=None, json=as_json)))


validator = _CoreValidator()
