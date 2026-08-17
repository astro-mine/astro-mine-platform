# SPDX-License-Identifier: Apache-2.0
"""Multi-fidelity profiles under one stable asset identity (RM-P0-FLEET-05).

An asset declares several **representations** — a cheap ``massmodel``, a ``kinematic``
model, a full ``articulated`` model — under one stable identity, so [Sim]'s scheduler can
dial fidelity per task without re-instantiating the asset (fleet.md §3/§11; conventions.md
§8: "Fleet declares the tiers, Sim chooses them"). Core owns the schema
(:class:`~astro_mine.core.sadf.model.FidelityProfile` and the ``FidelityTier`` enum); this
module is the authoring/selection API over it.

The ``surrogate`` tier — a *learned* substitute for expensive physics, owned by the
``Astro-Mine-Surrogate`` component — is **deferred to Phase 1** (RM-P0-FLEET-05 out of
scope): :func:`validate_profiles` rejects it rather than accept a profile whose tracked
error-bounds contract cannot yet be honored.

Backlog: RM-P0-FLEET-05 -- astro-mine-fleet#5
"""

from __future__ import annotations

from astro_mine.core.sadf.enums import FidelityTier
from astro_mine.core.sadf.model import Asset, FidelityProfile

__all__ = [
    "FIDELITY_ORDER",
    "FidelityError",
    "coarsest",
    "finest",
    "has_tier",
    "profiles",
    "select",
    "tiers",
    "validate_profiles",
]

#: The Phase-0 fidelity ladder, coarse → fine (cheapest first). ``surrogate`` is not a rung
#: on this ladder — it is an orthogonal, deferred tier (see module docstring).
FIDELITY_ORDER: tuple[FidelityTier, ...] = (
    FidelityTier.MASSMODEL,
    FidelityTier.KINEMATIC,
    FidelityTier.ARTICULATED,
)


class FidelityError(Exception):
    """An asset's fidelity profiles are not switchable under one identity."""


def validate_profiles(asset: Asset) -> None:
    """Assert the asset's fidelity profiles are well-formed and switchable.

    Raises :class:`FidelityError` if two profiles share a tier (the scheduler could not
    pick deterministically) or if a profile uses the deferred ``surrogate`` tier (or
    carries a surrogate descriptor). Zero or one profile is valid — a single-fidelity
    asset; the feature is only exercised once an asset declares two or more.
    """
    seen: set[FidelityTier] = set()
    for profile in asset.fidelity_profiles:
        tier = profile.tier
        if tier is FidelityTier.SURROGATE or profile.surrogate is not None:
            raise FidelityError(
                f"surrogate fidelity tiers are deferred to Phase 1 (RM-P0-FLEET-05 "
                f"out of scope); profile {tier.value!r} carries a surrogate tier/descriptor"
            )
        if tier in seen:
            raise FidelityError(f"duplicate fidelity tier {tier.value!r} under one identity")
        seen.add(tier)


def profiles(asset: Asset) -> list[FidelityProfile]:
    """The asset's declared fidelity profiles, validated and ordered coarse → fine."""
    validate_profiles(asset)
    return sorted(asset.fidelity_profiles, key=lambda p: FIDELITY_ORDER.index(p.tier))


def tiers(asset: Asset) -> list[FidelityTier]:
    """The declared tiers, ordered coarse → fine."""
    return [p.tier for p in profiles(asset)]


def has_tier(asset: Asset, tier: FidelityTier) -> bool:
    """Whether the asset declares a profile for ``tier``."""
    return any(p.tier is tier for p in asset.fidelity_profiles)


def select(asset: Asset, tier: FidelityTier) -> FidelityProfile:
    """The profile for ``tier`` — the representation the scheduler dials to, with the
    asset's identity unchanged. Raises :class:`FidelityError` if the tier is not declared.
    """
    for profile in profiles(asset):
        if profile.tier is tier:
            return profile
    raise FidelityError(
        f"asset {asset.identity.id!r} declares no {tier.value!r} fidelity profile "
        f"(has: {[t.value for t in tiers(asset)]})"
    )


def coarsest(asset: Asset) -> FidelityProfile:
    """The cheapest declared profile — the scheduler's default for large sweeps."""
    ordered = profiles(asset)
    if not ordered:
        raise FidelityError(f"asset {asset.identity.id!r} declares no fidelity profiles")
    return ordered[0]


def finest(asset: Asset) -> FidelityProfile:
    """The highest-fidelity declared profile — the validation-grade representation."""
    ordered = profiles(asset)
    if not ordered:
        raise FidelityError(f"asset {asset.identity.id!r} declares no fidelity profiles")
    return ordered[-1]
