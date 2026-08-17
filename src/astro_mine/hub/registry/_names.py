# SPDX-License-Identifier: Apache-2.0
"""The artifact-name rule, in one place (conventions.md §13, normative).

A published artifact's registry name is **bare kebab-case**: lowercase ASCII, hyphen-separated,
starting with a letter. No dots, no underscores, no uppercase, no slashes, and no package or
component prefix -- the name is ``prospecting-rover``, not ``astro-mine.fleet.prospecting-rover``
and not ``shackleton_water_ice_v1``.

**Why the rule exists at all.** Three conventions grew up side by side, one per producing component,
and they were all visible in a single catalog listing (G3.6): Fleet and Link published dotted
package-scoped names, Worlds and Surrogate published kebab, Prospect published snake with a version
suffix. A reader could not predict an artifact's name from what it is, which is the property a
registry name exists to have.

**A registry name is content identity, not an import path.** Which component produced an artifact is
a fact about its ``kind``, which Hub already carries as a first-class annotation. On a remote each
name is one repository -- ``ghcr.io/<prefix>/<name>:<version>`` -- which is exactly what makes the
dotted form read as the path component it is not, and the slashed form ambiguous with the prefix.

**The version lives in the tag, never in the name.** ``shackleton-de-gerlache:0.4.0``, not
``shackleton-de-gerlache-v1:0.4.0``. A name carrying its own ``-v1`` beside a SemVer tag states two
version numbers and says which one moves in neither.

**This is not the manifest's name.** :class:`~astro_mine.core.registry.PluginManifest` carries a
plugin identity, which is an import-ish address and legitimately dotted
(``astro-mine.mind.lawnmower-survey``). The two travel together through ``publish`` and are
different things; only the registry name is constrained here.

**Nothing calls this at publish yet, and that is deliberate.** The obvious home for it is
:meth:`HubClient.publish` — every publishing component reaches a registry through there. Wiring it
in was tried and reverted, because ``publish_asset`` passes ``name=identity.id``: a SADF asset's
authored id *is* its registry name, and all six shipped Fleet library assets carry legacy ids. A
gate would make ``astro-mine fleet publish`` refuse the platform's own documented examples, before
the migration that would fix them — and that migration is gated on the public flip (conventions.md
§13, "Artifact-name migration"), because renaming published content means re-publishing it under
new digests and keeping every existing scorecard resolvable.

So the rule ships as a **decision procedure** rather than a gate, in one place, with two callers
today and a third after the flip:

* ``tests/hub/test_artifact_names.py`` pins the exact legacy set. A *new* non-conforming name fails
  there immediately, which is what "new artifacts are born conformant" means while the old nine are
  still published under the names they were published under.
* the flip-time sweep, which needs exactly this predicate to know what it is renaming.
* ``HubClient.publish``, once that sweep has run and the shipped assets conform.

Legacy names are therefore **not errors** anywhere today: resolving, pulling, searching and scoring
a legacy artifact works exactly as before. A rule that broke the content it was written to tidy
would be the more expensive mistake.
"""

from __future__ import annotations

import re

__all__ = [
    "ARTIFACT_NAME_PATTERN",
    "InvalidArtifactName",
    "is_valid_artifact_name",
    "validate_artifact_name",
]

#: conventions.md §13. Anchored, so a conforming substring cannot smuggle a name through.
ARTIFACT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

#: The second half of the §13 rule, which the pattern above cannot express. A trailing ``-v<N>`` is
#: *valid kebab-case*, so `shackleton-de-gerlache-v1` passes the pattern while violating "the
#: version lives in the tag, never in the name" -- and that is one of the three legacy shapes this
#: gate exists to stop being minted. Checking the pattern alone would leave it born non-conformant.
#:
#: Anchored at the end and requiring digits, so `carbo-v` or `revision-vale` are untouched. A name
#: that genuinely ends in a version-shaped token is asking to be misread beside a SemVer tag.
_VERSION_SUFFIX = re.compile(r"-v\d+$")


class InvalidArtifactName(ValueError):
    """A name offered to ``publish`` that conventions.md §13 does not permit."""


def is_valid_artifact_name(name: str) -> bool:
    """Whether ``name`` conforms to the §13 artifact-name rule -- both halves of it."""
    return ARTIFACT_NAME_PATTERN.fullmatch(name) is not None and not _VERSION_SUFFIX.search(name)


def validate_artifact_name(name: str) -> str:
    """Return ``name`` if it conforms; raise :class:`InvalidArtifactName` naming the fix if not.

    The message says what to write instead, because the three legacy shapes each have an obvious
    correction and a reader who has just been refused wants that, not the regex.
    """
    if is_valid_artifact_name(name):
        return name

    suggestion = _suggest(name)
    detail = f" Did you mean {suggestion!r}?" if suggestion and suggestion != name else ""
    if ARTIFACT_NAME_PATTERN.fullmatch(name):
        # Shape is fine; it is the version suffix. Say only that, or the message sends the reader
        # hunting for a dot or an underscore that is not there.
        raise InvalidArtifactName(
            f"{name!r} is not a valid artifact name: it carries its own version suffix, and "
            f"conventions.md §13 puts the version in the tag rather than the name.{detail} "
            f"A name with a `-v1` beside a SemVer tag states two version numbers and says which "
            f"one moves in neither."
        )
    raise InvalidArtifactName(
        f"{name!r} is not a valid artifact name: conventions.md §13 requires bare kebab-case "
        f"(lowercase, hyphen-separated, starting with a letter; no dots, underscores, slashes or "
        f"uppercase, and no component prefix).{detail} A registry name is content identity, not an "
        f"import path."
    )


def _suggest(name: str) -> str | None:
    """A best-effort conforming form of ``name``, or ``None`` if nothing sensible falls out.

    Deliberately not a migration tool -- it exists to make the error message actionable. It strips a
    component prefix, flattens the separators the legacy shapes used, and drops a trailing version
    suffix, which between them cover every non-conforming name in the published anchor set.
    """
    candidate = name.lower()
    # `astro-mine.fleet.excavator` -> `excavator`: the prefix is a fact about `kind`, not the name.
    if "." in candidate:
        candidate = candidate.rsplit(".", 1)[-1]
    # `acme/no-entry` -> `no-entry`: a namespace is the `namespace=` argument, not part of the name.
    if "/" in candidate:
        candidate = candidate.rsplit("/", 1)[-1]
    candidate = candidate.replace("_", "-")
    # `shackleton-water-ice-v1` -> `shackleton-water-ice`: the version lives in the tag.
    candidate = re.sub(r"-v\d+$", "", candidate)
    candidate = re.sub(r"[^a-z0-9-]", "-", candidate).strip("-")
    candidate = re.sub(r"-{2,}", "-", candidate)
    return candidate if candidate and is_valid_artifact_name(candidate) else None
