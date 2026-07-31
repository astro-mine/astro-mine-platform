"""RM-P1-PROSPECT-11 — Hub-published community priors and prior-recipes (prospect.md §6, §12).

Proves the acceptance criterion "community priors / prior-recipes publish to and resolve from Hub as
content-addressed artifacts":

- **Community prior** — a contributed fitted prior publishes under the ``community`` namespace and
  reopens by digest through the *same* ``resource_field_backend`` entry point as the anchor prior.
- **Prior-recipe** — a :class:`PriorRecipeSpec` publishes as a content-addressed ``prior_recipe``
  artifact; resolving it **re-fits** and verifies the rebuilt prior reproduces the published content
  hash (recipe drift / non-reproduction fails closed).
- **Discover** — a registry's community field/recipe artifacts are enumerable.
- **Determinism** — the recipe spec is byte-stable, so its content address is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Registry, open_registry
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.publish import (
    PriorRecipeSpec,
    discover_priors,
    from_bundle,
    publish_community_prior,
    publish_recipe,
    recipe_reference_name,
    recipe_spec_from_bytes,
    resolve_recipe,
    serialize_recipe_spec,
)

_ANCHOR = "shackleton_water_ice_v1"
_PROBES = [(0.0, 0.0, 0.0), (500.0, -500.0, 0.0)]


def _small_grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _spec() -> PriorRecipeSpec:
    return PriorRecipeSpec(recipe=_ANCHOR, recipe_version="1.0.0", grid=_small_grid())


def _layers(registry: Registry, digest: str) -> dict[str, bytes]:
    image = registry.read_manifest(digest)
    return {layer["mediaType"]: registry.pull_blob(layer["digest"]) for layer in image["layers"]}


# --- community priors: publish under `community`, reopen by digest ----------------------------


def test_community_prior_publishes_and_reopens_by_digest(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    prior = load_prior(grid=_small_grid())
    artifact = publish_community_prior(
        prior,
        registry_path=tmp_path / "reg",
        publisher="an-external-lab",
        private_key_pem=private_pem,
    )

    consumer = HubClient(Registry(tmp_path / "reg"), trusted_public_key_pem=public_pem)
    assert consumer.verify(artifact.digest) == artifact.digest  # signed, verify-twice, fail-closed
    manifest = PluginManifest.model_validate_json(consumer.pull(artifact.digest))
    assert manifest.kind is PluginKind.RESOURCE_FIELD_BACKEND  # same contract as the anchor prior

    field = from_bundle(manifest, _layers(consumer.registry, artifact.digest))
    for probe in _PROBES:
        assert field.posterior(probe).mean == pytest.approx(prior.as_field().mean(probe))
        assert field.posterior(probe).variance > 0.0  # uncertainty-first, preserved


# --- prior-recipes: publish the recipe, resolve by re-fitting --------------------------------


def test_recipe_spec_is_byte_deterministic_and_content_addressed() -> None:
    assert serialize_recipe_spec(_spec()) == serialize_recipe_spec(_spec())
    assert _spec().content_hash == _spec().content_hash
    assert recipe_spec_from_bytes(serialize_recipe_spec(_spec())) == _spec()


def test_recipe_publishes_and_resolves_by_refitting(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    spec = _spec()
    artifact = publish_recipe(
        spec,
        registry_path=tmp_path / "reg",
        publisher="an-external-lab",
        private_key_pem=private_pem,
    )
    assert artifact.reference == f"{recipe_reference_name(_ANCHOR)}:1.0.0"

    rebuilt = resolve_recipe(
        artifact.reference, registry_path=tmp_path / "reg", trusted_public_key_pem=public_pem
    )
    # The recipe re-fits the same prior the spec's author fitted (content hash matches).
    assert isinstance(rebuilt, Prior)
    reference = spec.build()
    assert rebuilt.content_hash == reference.content_hash


def test_resolve_recipe_rejects_a_prior_reference(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    # A resource_field_backend (fitted-field) artifact is not a recipe — resolving it as one fails.
    artifact = publish_community_prior(
        load_prior(grid=_small_grid()),
        registry_path=tmp_path / "reg",
        publisher="lab",
        private_key_pem=private_pem,
    )
    with pytest.raises(ValueError, match="not a prior_recipe"):
        resolve_recipe(artifact.reference, registry_path=tmp_path / "reg")


def test_resolve_recipe_fails_closed_on_reproduction_drift(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    spec = _spec()
    publish_recipe(
        spec, registry_path=tmp_path / "reg", publisher="lab", private_key_pem=private_pem
    )

    # Tamper with the published prior_content_hash so the rebuilt prior no longer matches it.
    registry = Registry(tmp_path / "reg")
    reference = f"{recipe_reference_name(_ANCHOR)}:1.0.0"
    descriptor = registry.resolve(reference)
    manifest = PluginManifest.model_validate_json(registry.read_config(descriptor.digest))
    manifest.attributes["prior_content_hash"] = "deadbeef"
    # Re-publish the tampered manifest to a fresh registry and try to resolve it.
    poisoned = tmp_path / "poisoned"
    layers = _layers(registry, descriptor.digest)
    from astro_mine.hub.registry import Blob

    Registry(poisoned).publish(
        name=recipe_reference_name(_ANCHOR),
        version="1.0.0",
        kind="plugin",
        config=manifest.model_dump_json().encode("utf-8"),
        layers=[Blob(mt, data) for mt, data in layers.items()],
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        resolve_recipe(reference, registry_path=poisoned)


# --- discovery -------------------------------------------------------------------------------


def test_discover_lists_community_priors_and_recipes(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    reg = tmp_path / "reg"
    publish_community_prior(
        load_prior(grid=_small_grid()),
        registry_path=reg,
        publisher="lab-a",
        private_key_pem=private_pem,
    )
    publish_recipe(_spec(), registry_path=reg, publisher="lab-b", private_key_pem=private_pem)

    found = discover_priors(open_registry(str(reg)))
    kinds = {a.kind for a in found}
    assert "resource_field_backend" in kinds
    assert "prior_recipe" in kinds
    assert all(a.species for a in found)  # every discovered artifact carries its species facet
