# SPDX-License-Identifier: Apache-2.0
"""A release tag must name the version the tree declares (astro-mine-platform#33).

`VERSIONING.md` §2.1 is blunt about why this exists. The platform's version used to be *derived*
from the Git tag by `hatch-vcs`, which made "bump the version" and "cut the tag" the same act — they
could not drift because they were one thing. Moving to **maturin** (Guard's Rust core has to be
compiled into the wheel) took `hatch-vcs` with it, so the version is now a static string in
`pyproject.toml` and the two are separate acts that a human must keep in step.

§2.1 calls that "the weaker half of the scheme" and asks for a check to replace the guarantee:

> a release **MUST** bump the static version and cut the tag **in the same commit**, and CI
> **SHOULD** fail a tag whose name disagrees with the declared version.

This is that check. It lives in `tests/platform/` with the repository's other gates
(`test_typecheck_ratchet`, `test_layering`, `test_licence_headers`) rather than only in a workflow,
for a reason that is specific rather than stylistic: the org's Actions minutes are exhausted
(astro-mine/.github#8), so a check that existed only as a workflow step could not have been observed
running before it was relied on — and this one guards the *first tag this organisation has ever
cut*. A gate nobody has watched work is not a gate.

**The rule is tested on synthetic input as well as on the tree**, because until the first tag exists
the repository-level assertion has nothing to look at, and a check that passes because it found
nothing to check is the failure mode this file is trying to prevent elsewhere.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import subprocess
import tomllib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DISTRIBUTION = "astro-mine-platform"


def declared_version() -> str:
    """The static version in ``pyproject.toml`` — the one authority now that hatch-vcs is gone."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def tag_disagrees(tag: str, version: str) -> bool:
    """Whether release ``tag`` fails to name ``version``. The rule, in one place.

    A release tag is ``v`` followed by the declared version, exactly. Anything else — a missing
    prefix, a stale number, a suffix the version does not carry — is a disagreement, because the tag
    is what a consumer resolves and the version is what the wheel reports, and a release where those
    differ is a release nobody can pin correctly.
    """
    return tag != f"v{version}"


def _tags_on_head() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "--points-at", "HEAD", "--list", "v*"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_the_packaged_metadata_matches_the_declared_version() -> None:
    """The installed distribution reports what ``pyproject.toml`` declares.

    Unconditional, and the half that catches an editable install left behind by a bump — which is
    the same drift class the tag check covers, one step earlier.
    """
    assert importlib.metadata.version(DISTRIBUTION) == declared_version()


@pytest.mark.parametrize(
    ("tag", "version", "disagrees"),
    [
        ("v0.1.0", "0.1.0", False),
        ("v0.2.0", "0.1.0", True),  # the stale-number case §2.1 is actually worried about
        ("0.1.0", "0.1.0", True),  # no `v` prefix
        ("v0.1.0-rc1", "0.1.0", True),  # a suffix the declared version does not carry
        ("v0.1", "0.1.0", True),  # truncated
    ],
)
def test_the_rule_rejects_a_disagreeing_tag(tag: str, version: str, disagrees: bool) -> None:
    """The rule itself, on synthetic input — meaningful before any tag exists."""
    assert tag_disagrees(tag, version) is disagrees


def test_a_release_tag_on_this_commit_names_the_declared_version() -> None:
    """The rule applied to the tree. Silent when this commit is not a release, which is correct.

    Cutting a tag is the act this guards, so there is nothing to say about a commit that is not one.
    The rule's logic is covered above regardless, so this cannot pass merely because it found
    nothing.
    """
    version = declared_version()
    offenders = [tag for tag in _tags_on_head() if tag_disagrees(tag, version)]
    assert offenders == [], (
        f"release tag(s) {offenders} on this commit do not name the declared version {version!r}. "
        f"VERSIONING.md §2.1 requires the static version bump and the tag in the same commit — "
        f"either retag as v{version}, or bump pyproject.toml to match."
    )
