"""Hub-published community priors and prior-recipes (RM-P1-PROSPECT-11; prospect.md §6, §12).

Two content-addressed community-contribution paths on top of the anchor publish flow
(:mod:`astro_mine.prospect.publish._publish`):

- **Community priors** — :func:`publish_community_prior` publishes any fitted :class:`Prior` under a
  ``community`` Hub namespace, so a contributed prior is discoverable and reusable *by digest*
  alongside the org's anchor prior, resolved through the exact same ``resource_field_backend``
  entry point (no new consumer code).
- **Prior-recipes** — a :class:`PriorRecipeSpec` names a registered recipe + the grid it fits over.
  :func:`publish_recipe` serializes it to a content-addressed artifact (Core ``prior_recipe`` kind)
  and :func:`resolve_recipe` pulls it back and **re-fits**, verifying the rebuilt prior reproduces
  the published content hash — so the recipe, not just the frozen field, is the shareable,
  reproducible unit (prospect.md §3 "prior recipes … contributed as plugins and indexed by Hub").

:func:`discover_priors` lists the community field/recipe artifacts a local registry holds — the
"discover" half of publish/discover.

Backlog: RM-P1-PROSPECT-11 — https://github.com/astro-mine/astro-mine-prospect/issues/21
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.prospect import __version__ as _PROSPECT_VERSION
from astro_mine.prospect.field.metadata import FieldGrid
from astro_mine.prospect.priors.recipe import Prior, get_recipe
from astro_mine.prospect.publish._bundle import bundle_digest
from astro_mine.prospect.publish._publish import publish_prior

if TYPE_CHECKING:
    from astro_mine.hub.registry import PublishedArtifact, RegistryClient

__all__ = [
    "RECIPE_SPEC_MEDIA_TYPE",
    "DiscoveredArtifact",
    "PriorRecipeSpec",
    "build_recipe_manifest",
    "discover_priors",
    "publish_community_prior",
    "publish_recipe",
    "recipe_reference_name",
    "recipe_spec_from_bytes",
    "resolve_recipe",
    "serialize_recipe_spec",
]

#: The Hub namespace community contributions publish under (distinct from the org's ``open``
#: anchor).
COMMUNITY_NAMESPACE = "community"

#: The single OCI layer's media type — the canonical prior-recipe spec JSON.
RECIPE_SPEC_MEDIA_TYPE = "application/vnd.astro-mine.prior-recipe.spec.v1+json"

#: Recipe artifacts get a distinct reference name so a recipe and its fitted prior can coexist in
#: one
#: registry (both are Hub ``plugin`` artifacts; the Core manifest kind — field vs recipe — differs).
_RECIPE_NAME_SUFFIX = "-recipe"


def recipe_reference_name(recipe: str) -> str:
    """The Hub artifact name a prior-recipe publishes under (distinct from its fitted-prior
    name)."""
    return f"{recipe}{_RECIPE_NAME_SUFFIX}"


#: The kinds :func:`discover_priors` surfaces — a published field and a published recipe.
_DISCOVERABLE_KINDS = frozenset({PluginKind.RESOURCE_FIELD_BACKEND, PluginKind.PRIOR_RECIPE})


class PriorRecipeSpec(BaseModel):
    """A portable, content-addressed recipe for refitting a prior: a registered recipe + its grid.

    Publishing the *recipe* (not just a frozen field) lets a community member reproduce or re-fit
    the
    prior over their own grid. Resolving requires the named recipe to be **registered** (recipes are
    plugins, prospect.md §3); :meth:`build` re-runs it and checks the installed recipe's version
    matches, so recipe drift fails loudly rather than silently returning a different prior.

    Frozen and content-addressable — :attr:`content_hash` is the recipe artifact's identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe: str
    recipe_version: str
    grid: FieldGrid

    @property
    def content_hash(self) -> str:
        """A stable SHA-256 over the canonicalized spec — the recipe artifact's content address."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def build(self) -> Prior:
        """Re-fit the prior from the registered recipe over :attr:`grid` (fails on recipe drift).

        Raises ``ValueError`` if the recipe is not registered, or if the installed recipe fits a
        different version than the spec pins — either would silently change the prior.
        """
        prior = get_recipe(self.recipe)(self.grid)
        built_version = prior.provenance.recipe_version
        if built_version != self.recipe_version:
            raise ValueError(
                f"recipe {self.recipe!r} is registered at version {built_version!r} but the spec "
                f"pins {self.recipe_version!r}; the recipe drifted — refusing to rebuild"
            )
        return prior


def serialize_recipe_spec(spec: PriorRecipeSpec) -> bytes:
    """Serialize *spec* to canonical JSON bytes (sorted keys, no whitespace) — a stable content
    address."""
    return json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def recipe_spec_from_bytes(data: bytes) -> PriorRecipeSpec:
    """Rebuild a :class:`PriorRecipeSpec` from its canonical JSON bytes."""
    return PriorRecipeSpec.model_validate_json(data)


def build_recipe_manifest(
    spec: PriorRecipeSpec, prior: Prior, *, spec_sha256: str
) -> PluginManifest:
    """Build the Core ``prior_recipe`` manifest for *spec* (unsigned; the caller attaches the
    signature).

    ``spec_sha256`` is the ``sha256:<hex>`` of the serialized spec layer; ``prior`` is the spec's
    freshly-built prior, whose content hash is folded into ``attributes`` as ``prior_content_hash``
    so
    :func:`resolve_recipe` can verify the recipe reproduces the same fitted field. The manifest's
    ``provenance.digest`` is the spec's own content address (the recipe artifact's identity).
    """
    return PluginManifest(
        name=spec.recipe,
        version=spec.recipe_version,
        kind=PluginKind.PRIOR_RECIPE,
        license="Apache-2.0",
        description=(
            f"Prior-recipe {spec.recipe!r} v{spec.recipe_version} for {prior.metadata.species} "
            f"({prior.metadata.unit}) — refits the prior over the pinned grid."
        ),
        provenance=Provenance(
            digest=f"sha256:{spec.content_hash}",
            code_version=spec.recipe_version,
            toolchain_version=f"astro-mine-prospect {_PROSPECT_VERSION}",
        ),
        attributes={
            "species": prior.metadata.species,
            "unit": prior.metadata.unit,
            "recipe": spec.recipe,
            "recipe_version": spec.recipe_version,
            "spec_media_type": RECIPE_SPEC_MEDIA_TYPE,
            "spec_sha256": spec_sha256,
            "prior_content_hash": prior.content_hash,
        },
    )


def publish_community_prior(
    prior: Prior,
    *,
    registry: RegistryClient,
    publisher: str,
    private_key_pem: bytes,
    name: str | None = None,
    version: str | None = None,
) -> PublishedArtifact:
    """Publish a community-contributed *prior* under the ``community`` namespace, by digest.

    A thin wrapper over :func:`~astro_mine.prospect.publish._publish.publish_prior` that stamps the
    contribution with its ``publisher`` and the ``community`` Hub namespace, so it is discoverable
    and
    reusable alongside the org anchor prior *without a distinct resolve path* — a consumer reopens
    it
    through the same ``resource_field_backend`` entry point. ``publisher`` is required (a community
    prior is attributed).
    """
    return publish_prior(
        prior,
        registry=registry,
        private_key_pem=private_key_pem,
        name=name,
        version=version,
        publisher=publisher,
        namespace=COMMUNITY_NAMESPACE,
    )


def publish_recipe(
    spec: PriorRecipeSpec,
    *,
    registry: RegistryClient,
    publisher: str,
    private_key_pem: bytes,
    namespace: str = COMMUNITY_NAMESPACE,
) -> PublishedArtifact:
    """Serialize, manifest, sign, and publish a prior-*recipe* to the local OCI registry, by digest.

    Builds the prior once (validating the recipe is registered and un-drifted), serializes the spec,
    and publishes it as a Core ``prior_recipe`` artifact whose manifest records the prior's content
    hash — so :func:`resolve_recipe` can prove the recipe reproduces the same fitted field.
    """
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.registry import Blob

    prior = spec.build()
    spec_bytes = serialize_recipe_spec(spec)
    manifest = build_recipe_manifest(spec, prior, spec_sha256=bundle_digest(spec_bytes))
    client = HubClient(registry)
    return client.publish(
        name=recipe_reference_name(spec.recipe),
        version=spec.recipe_version,
        kind="plugin",  # the Hub artifact category; the Core manifest kind is PRIOR_RECIPE
        manifest=manifest,
        layers=[Blob(RECIPE_SPEC_MEDIA_TYPE, spec_bytes)],
        private_key_pem=private_key_pem,
        namespace=namespace,
        publisher=publisher,
    )


def resolve_recipe(
    reference: str,
    *,
    registry: RegistryClient,
    trusted_public_key_pem: bytes | None = None,
) -> Prior:
    """Pull a published prior-recipe by ``reference`` and re-fit it, fail-closed on reproduction
    drift.

    Verifies the artifact (verify-twice, signed when a trusted key is given), reads the spec layer,
    checks the spec bytes match the manifest's ``spec_sha256`` (integrity), rebuilds the prior, and
    checks the rebuilt prior's content hash matches the published ``prior_content_hash``. Any
    mismatch
    raises ``ValueError`` — a recipe that does not reproduce its published field is rejected.
    """
    from astro_mine.hub.client import HubClient

    client = HubClient(registry, trusted_public_key_pem=trusted_public_key_pem)
    # Require the full signed supply chain when a trusted key is given (a signed contribution);
    # otherwise verify integrity/digest only, so an unsigned community recipe still resolves.
    require = ("signature", "slsa", "sbom") if trusted_public_key_pem is not None else ()
    manifest = PluginManifest.model_validate_json(client.pull(reference, require=require))
    if manifest.kind is not PluginKind.PRIOR_RECIPE:
        raise ValueError(f"{reference!r} is a {manifest.kind.value} artifact, not a prior_recipe")

    descriptor = registry.resolve(reference)
    image = registry.read_manifest(descriptor.digest)
    layers = {layer["mediaType"]: registry.pull_blob(layer["digest"]) for layer in image["layers"]}
    try:
        spec_bytes = layers[RECIPE_SPEC_MEDIA_TYPE]
    except KeyError:
        raise ValueError(
            f"prior-recipe artifact {reference!r} has no {RECIPE_SPEC_MEDIA_TYPE!r} layer"
        ) from None
    if bundle_digest(spec_bytes) != manifest.attributes["spec_sha256"]:
        raise ValueError(
            f"prior-recipe spec bytes do not match the manifest digest for {reference!r}"
        )

    prior = recipe_spec_from_bytes(spec_bytes).build()
    expected = manifest.attributes["prior_content_hash"]
    if prior.content_hash != expected:
        raise ValueError(
            f"recipe {reference!r} rebuilt a prior with hash {prior.content_hash} != published "
            f"{expected}: the recipe does not reproduce its field"
        )
    return prior


@dataclass(frozen=True)
class DiscoveredArtifact:
    """A community prior or recipe found in a registry: its reference, digest, kind, and facets."""

    reference: str
    digest: str
    kind: str
    species: str
    recipe: str


def discover_priors(registry: RegistryClient) -> tuple[DiscoveredArtifact, ...]:
    """List the community prior fields and prior-recipes a local registry holds (sorted by
    reference).

    The discover half of publish/discover: walks the registry's references, reads each Core
    manifest,
    and surfaces the ``resource_field_backend`` (fitted fields) and ``prior_recipe`` (recipes)
    artifacts with their species/recipe facets, so a consumer can pick one to resolve by digest.
    """

    found: list[DiscoveredArtifact] = []
    for reference in registry.references():
        descriptor = registry.resolve(reference)
        manifest = PluginManifest.model_validate_json(registry.read_config(descriptor.digest))
        if manifest.kind not in _DISCOVERABLE_KINDS:
            continue
        found.append(
            DiscoveredArtifact(
                reference=reference,
                digest=descriptor.digest,
                kind=manifest.kind.value,
                species=str(manifest.attributes.get("species", "")),
                recipe=str(manifest.attributes.get("recipe", "")),
            )
        )
    return tuple(sorted(found, key=lambda a: a.reference))
