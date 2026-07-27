"""The static first-party verb manifest — plain strings, no imports, no dependencies.

Discovery alone (:mod:`astro_mine.cli._discovery`) cannot say anything about a component that is
**not installed**: with no ``astro-mine-learn`` there is no ``train`` entry point, and the best a
purely dynamic umbrella could manage is *"unknown command"* — which tells a user nothing about a
platform they are still learning the shape of. This table is the minimal fix (RFC-0011 §1b): it
maps the platform's own verbs to the distribution that provides each, so a missing component
produces *"`astro-mine train` needs astro-mine-learn — pip install astro-mine-learn"*.

It does two jobs, and both are why it stays strings:

1. **The install hint**, above.
2. **Top-level help text.** Listing a one-line description next to each verb would otherwise mean
   loading every provider to read its ``help`` — exactly the import-everything cost RFC-0011 §1a
   forbids. Taking first-party help from this table keeps ``astro-mine --help`` free. A verb's
   *complete* help still comes from the provider, on ``astro-mine <verb> --help``, where paying
   for one import is what the user asked for.

**It governs first-party verbs only.** A third-party verb is discovered dynamically, listed with
its providing distribution, and needs no entry here — the no-PR-to-extend rule (RFC-0011 §3) is
not quietly reintroduced through this file.

Entries are added when a component actually registers the verb, or ahead of it as a promise the
umbrella can keep honestly ("not installed"), never as a claim that it works.

``validate`` is deliberately **absent**: the umbrella owns that verb itself (RFC-0011 §6), so it
can never be the missing-component case this table exists to describe. Its own error names the
package that owns the format at hand — which is more specific than anything a static row could
say, since `validate` has several owners. ``new`` and ``plugin`` are absent for the same reason —
but the *kinds* they route to are components' own, so those get tables of their own below.

**This module imports nothing and computes nothing.** It is data plus string formatting, which is
what lets `astro-mine --help` stay free. In particular the *"installed but registers no such
kind"* case is not decided here: that needs an ``importlib.metadata`` probe, and it lives with the
verb that needs it (:mod:`astro_mine.cli._new`).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import NamedTuple

__all__ = [
    "FIRST_PARTY_KINDS",
    "FIRST_PARTY_PLUGIN_KINDS",
    "FIRST_PARTY_VERBS",
    "FirstPartyKind",
    "FirstPartyVerb",
    "install_hint",
]


class FirstPartyVerb(NamedTuple):
    """The distribution that provides a platform verb, and one line about what it does."""

    distribution: str
    help: str


#: Platform verb → provider. Ordered as a user meets them: get content, run it, score it, train,
#: publish, then the component-scoped surfaces. `astro-mine <component> <verb>` is RFC-0011 §2's
#: form for actions that are inherently component-scoped, so those components appear here as
#: single verbs (`studio`, `fleet`, …) rather than exploding their whole subcommand tree.
FIRST_PARTY_VERBS: MappingProxyType[str, FirstPartyVerb] = MappingProxyType(
    {
        "fetch": FirstPartyVerb("astro-mine-bench", "download a scenario's pinned content"),
        "list": FirstPartyVerb("astro-mine-bench", "list the scenarios in the zoo"),
        "score": FirstPartyVerb("astro-mine-bench", "run a policy on a scenario and score it"),
        "submit": FirstPartyVerb("astro-mine-bench", "submit a policy to a leaderboard"),
        "run": FirstPartyVerb("astro-mine-sim", "run a scenario in the simulator"),
        "record": FirstPartyVerb("astro-mine-sim", "record a self-contained Sim scenario file"),
        "train": FirstPartyVerb("astro-mine-learn", "train a policy and export it"),
        "publish": FirstPartyVerb("astro-mine-hub", "publish a signed artifact to a registry"),
        "search": FirstPartyVerb("astro-mine-hub", "discover artifacts in a registry"),
        "pull": FirstPartyVerb("astro-mine-hub", "pull and re-verify an artifact"),
        "verify": FirstPartyVerb("astro-mine-hub", "re-verify an artifact's supply chain"),
        "studio": FirstPartyVerb("astro-mine-studio", "the design studio (`studio serve`)"),
        "fleet": FirstPartyVerb("astro-mine-fleet", "author and publish SADF assets"),
        "worlds": FirstPartyVerb("astro-mine-worlds", "build and publish world bundles"),
        "prospect": FirstPartyVerb("astro-mine-prospect", "publish resource priors"),
        "link": FirstPartyVerb("astro-mine-link", "publish contact plans"),
        "mind": FirstPartyVerb("astro-mine-mind", "validate and compose planner stacks"),
        "guard": FirstPartyVerb("astro-mine-guard", "author, compile and falsify SafetySpecs"),
        "cloud": FirstPartyVerb("astro-mine-cloud", "submit and manage cluster jobs"),
    }
)


class FirstPartyKind(NamedTuple):
    """The distribution that owns a scaffold kind, and one line about what it writes."""

    distribution: str
    help: str


#: Authored-document kinds → owner (`astro-mine new <kind>`; RFC-0011 §7). Every one of them is
#: registered by its owner now that Worlds ships a `WorldSpec` example, a validator and a scaffold
#: (G2.11, astro-mine/astro-mine-worlds#57) — so this table is once again a description of what the
#: platform offers rather than a promise about part of it.
FIRST_PARTY_KINDS: MappingProxyType[str, FirstPartyKind] = MappingProxyType(
    {
        "asset": FirstPartyKind("astro-mine-fleet", "a SADF asset (the exemplar: `fleet new`)"),
        "stack": FirstPartyKind("astro-mine-mind", "an autonomy stack spec"),
        "safety": FirstPartyKind("astro-mine-guard", "a SafetySpec"),
        "world": FirstPartyKind("astro-mine-worlds", "a WorldSpec"),
    }
)

#: Plugin kinds → the distribution that hosts the extension group each is written against
#: (`astro-mine plugin new <kind>`). The kinds are the live entry-point groups documented in the
#: platform's plugin-authoring guide (`guide/how-to/write-a-plugin.md`, G2.8), which is the
#: authority on what each scaffold must emit.
#:
#: ``cli`` is **absent**, exactly as ``validate`` is absent above: the umbrella owns the
#: ``astro_mine.cli`` group, so it owns that scaffold and it can never be a missing component.
FIRST_PARTY_PLUGIN_KINDS: MappingProxyType[str, FirstPartyKind] = MappingProxyType(
    {
        "tier": FirstPartyKind(
            "astro-mine-mind", "an autonomy tier (astro_mine.mind.tier_plugins)"
        ),
        "provider": FirstPartyKind("astro-mine-sim", "a content provider (astro_mine.providers)"),
        "field-model": FirstPartyKind(
            "astro-mine-worlds", "an illumination backend (astro_mine.field_models)"
        ),
        "runner": FirstPartyKind(
            "astro-mine-bench", "a Bench execution backend (astro_mine.bench.runners)"
        ),
        "solver": FirstPartyKind(
            "astro-mine-allocate", "an allocation backend (astro_mine.allocate.solvers)"
        ),
        "algorithm": FirstPartyKind(
            "astro-mine-learn", "a MARL algorithm (astro_mine.learn.algorithms)"
        ),
        "curriculum": FirstPartyKind(
            "astro-mine-learn", "a training curriculum (astro_mine.learn.curricula)"
        ),
    }
)


def install_hint(verb: str) -> str | None:
    """The one-line fix for a known verb whose component is not installed.

    ``None`` for a verb this table does not know — that case is an unknown-command error listing
    what *is* available, not a fabricated install suggestion.
    """
    known = FIRST_PARTY_VERBS.get(verb)
    if known is None:
        return None
    return (
        f"`astro-mine {verb}` needs {known.distribution} — "
        f"install it with `pip install {known.distribution}` "
        f"(or `uv add {known.distribution}`), then re-run."
    )
