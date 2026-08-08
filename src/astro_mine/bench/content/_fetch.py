"""Populate a local OCI-layout registry from the published anchor content (G1.2; bench#56).

A user who clones Bench can list scenarios and score a fixture, and — until this command — had no
way to obtain the content the anchor pins. :func:`fetch_scenario_content` closes that: it resolves a
:class:`~astro_mine.bench.scenario.ScenarioSpec`'s pins, mirrors each **by digest** from the
published registry into a local OCI-layout store, and leaves a store the Sim-backed runner can read
offline (``score --runner sim``, ``astro-mine sim run``).

**Mirroring, not re-publishing.** Hub exposes no registry-to-registry copy, and re-publishing an
artifact from *source* re-stamps its manifest and changes its digest — which is how the anchor pins
drifted on the last re-pin. So each pin is reconstructed from its own verified bytes: read the
source manifest, pull the config and layer blobs (the remote transport re-hashes every one before it
returns), then re-publish locally with byte-identical config, layers, media types and annotations.
``Registry.publish`` rebuilds the image manifest deterministically through ``canonical_manifest``,
so feeding back identical bytes reproduces the identical digest — and this module **asserts** that
rather than assuming it (:class:`DigestMismatch`), because a mirror that silently changes a digest
would break every pin that references it.

**Verify-twice, fail-closed** (hub.md §2.3, §9; conventions.md §9). Bytes are re-hashed on the way
out of the source, the reconstructed digest must equal the pinned digest, the attestations (cosign
signature, SLSA provenance, SBOM) are mirrored through OCI Referrers, and the finished local
artifact is re-verified through :class:`~astro_mine.hub.client.HubClient` before the pin counts as
fetched. Any failure aborts; nothing partial is left resolvable.

**The trust anchor is optional and its absence is stated, not hidden** (bench#56 D6). The nine
digests arrive inside this wheel (``zoo/<scenario>/scenario.json``), not from the registry, so
content addressing — not signer pinning — is what makes a substituted artifact impossible: any
substitution changes the hash and fails here. ``trusted_key_pem`` additionally pins *whose*
signature it is; without it a signature must still be present, intact and bound to the subject, but
any key satisfies it. A caller deciding trust from an *untrusted* reference (a leaderboard
submission) is a different gate and must pass a key — see ``astro_mine.seal._supply_chain.verify``.

Bench never imports Sim (conventions.md §1.1; bench.md §2.2): this module writes a store and returns
a **path**. Resolving that path into a live bundle store is Sim's job, through its own public
``open_bundle_store``. The Hub client is imported lazily behind the ``[fetch]`` extra so the base
package keeps importing on ``core + pydantic`` alone.

Backlog: bench#56 — astro-mine-bench#56
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astro_mine.bench.scenario import ScenarioSpec

__all__ = [
    "DEFAULT_CONTENT_SOURCE",
    "STORE_ENV",
    "DigestMismatch",
    "FetchError",
    "FetchedPin",
    "default_store_path",
    "fetch_scenario_content",
    "resolve_store_path",
]

#: Where the anchor content set is published (astro-mine-hub `docs/publishing-the-anchor-content-
#: set.md`). Nothing machine-readable names it — `scenario.json` has no registry field (its
#: `"registry"` key is a *Core interface version*, not a location) and `pins.json` carries `<reg>`
#: placeholders — so the default lives here, overridable with `--from`. GHCR packages stay private
#: until the org flips public: a pull needs `$GITHUB_TOKEN` with `read:packages` until then, which
#: Hub's GHCR auth picks up on its own.
DEFAULT_CONTENT_SOURCE = "ghcr.io/astro-mine"

#: The local content store, shared with `astro-mine sim run` and the `sim` Bench runner
#: (astro-mine-sim `sim/bench/_runner.py`). One convention across both CLIs.
STORE_ENV = "ASTRO_MINE_HUB_REGISTRY"

_INSTALL_HINT = (
    "`fetch` needs the Hub client, which is not in the base package: install it with "
    "`uv sync --extra fetch` (or `pip install 'astro-mine-platform[bench-fetch]'`)"
)


class FetchError(Exception):
    """A pin could not be fetched — always fail closed, never a partial success."""


class DigestMismatch(FetchError):
    """A mirrored artifact did not reproduce its pinned digest.

    The mirror is only sound because re-publishing identical bytes rebuilds an identical manifest.
    If that ever stops holding — a Hub manifest-shape change, an annotation Bench did not carry
    across — every downstream reference to this pin would break. Fail here, loudly, instead.
    """


@dataclass(frozen=True, slots=True)
class FetchedPin:
    """One pin's outcome: what it is, where it landed, and whether this run moved bytes."""

    content_id: str
    digest: str
    reference: str
    #: ``False`` when the pin was already present and re-verified — the idempotent offline path.
    mirrored: bool
    size_bytes: int


def default_store_path() -> Path:
    """The default local content store — an XDG-style cache dir.

    Deliberately *not* ``files/hub-registry``: that is a development-workspace convention, and
    teaching it as a product default is what left Sim's own docstring pointing at a path that ships
    nowhere (bench#56 D5).
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "astro-mine" / "hub-registry"


def resolve_store_path(explicit: str | Path | None = None) -> Path:
    """The store to populate: ``--registry`` > ``$ASTRO_MINE_HUB_REGISTRY`` > the XDG default."""
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(STORE_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return default_store_path()


def _artifact_kind(artifact_type: str) -> str:
    """The Hub artifact kind whose media type is ``artifact_type``.

    Recovered from the manifest's ``artifactType`` rather than from the Core manifest's own ``kind``
    field, because **the two vocabularies are not the same** (astro-mine-hub#33): a Worlds bundle is
    Hub kind ``world`` but Core kind ``world_provider``, and a Prospect prior is Hub ``plugin`` but
    Core ``resource_field_backend``. Publishing under the Core kind would emit a different
    ``artifactType``, change the manifest, and break the digest.
    """
    from astro_mine.hub.registry import ARTIFACT_KINDS, artifact_media_type

    for kind in ARTIFACT_KINDS:
        if artifact_media_type(kind) == artifact_type:
            return kind
    raise FetchError(f"unknown artifact type {artifact_type!r}; no Hub artifact kind produces it")


def _identity(config_bytes: bytes) -> tuple[str, str]:
    """The ``(name, version)`` an artifact was published under, from its Core plugin manifest.

    Taken from the artifact's own verified config blob so it works identically against a local and a
    remote source — a remote registry does not expose the local layout's index annotations.

    **The config blob is a bare** ``PluginManifest`` — ``name`` at the top level (hub.md §2
    principle 2). This function nevertheless also accepts a ``ManifestDocument`` envelope
    (``{"manifest_version": …, "manifest": {…}}``), and the reason is specific to *mirroring*
    rather than a hedge about which convention is right.

    It used to be a hedge. This docstring recorded "two config shapes ship" as though the platform
    had two conventions, which is how astro-mine-platform#14 stayed invisible: the disagreement had
    been *found* here and absorbed instead of fixed, so nothing failed until Bench's intake met a
    real artifact. It has one convention now, and the intake gates enforce it
    (:func:`~astro_mine.core.registry.load_plugin_manifest`).

    The leniency stays here because a mirror is not a reader. It copies bytes it does not
    interpret, and it needs from the config only the ``(name, version)`` to re-publish under —
    never the manifest's meaning. Refusing to mirror a remote or legacy artifact over the shape of
    a field it is merely transcribing would break ``astro-mine bench fetch`` against content this
    platform did not publish, and would buy nothing: whatever is wrong with such an artifact is
    caught when something actually tries to *use* it.
    """
    try:
        document = json.loads(config_bytes)
    except ValueError as exc:
        raise FetchError(f"artifact config is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise FetchError("artifact config is not a JSON object")
    inner = document.get("manifest")
    manifest = inner if isinstance(inner, dict) else document
    try:
        return str(manifest["name"]), str(manifest["version"])
    except (KeyError, TypeError) as exc:
        raise FetchError(
            f"artifact config carries no Core plugin manifest identity: missing {exc}"
        ) from exc


def _mirror_referrers(source: Any, dest: Any, digest: str, blob_cls: type) -> None:
    """Copy an artifact's attestations (signature, SLSA provenance, SBOM) across.

    The referrer *manifests* are rebuilt by ``attach`` rather than byte-copied — nothing pins their
    digests, and only the subject digest and the attestation payload are load-bearing for
    verification.
    """
    for descriptor in source.referrers(digest):
        referrer = source.read_manifest(descriptor.digest)
        layers = referrer.get("layers") or []
        if len(layers) != 1:
            raise FetchError(
                f"attestation {descriptor.digest} on {digest} carries {len(layers)} layers; "
                "expected exactly one"
            )
        layer = layers[0]
        dest.attach(
            subject=digest,
            artifact_type=referrer["artifactType"],
            blob=blob_cls(layer["mediaType"], source.pull_blob(layer["digest"])),
            annotations=referrer.get("annotations"),
        )


def _already_present(client: Any, digest: str) -> bool:
    """True when ``digest`` is already in the local store *and* still verifies fail-closed.

    Presence alone is not enough: a half-written or tampered store must re-fetch rather than be
    trusted for being on disk. This is what makes a second ``fetch`` safe offline (CX-LOCAL).
    """
    try:
        client.verify(digest)
    except Exception:
        return False
    return True


def fetch_scenario_content(
    spec: ScenarioSpec,
    *,
    source: str | Path = DEFAULT_CONTENT_SOURCE,
    store: str | Path | None = None,
    trusted_key_pem: bytes | None = None,
    on_event: Callable[[str], None] | None = None,
) -> tuple[FetchedPin, ...]:
    """Mirror every pin ``spec`` declares into a local OCI-layout store; return what landed.

    ``source`` is the published registry (a remote URL or a local layout — ``open_registry``
    dispatches). ``store`` defaults through :func:`resolve_store_path`. ``trusted_key_pem`` pins the
    signer when supplied; see this module's docstring for why it is optional here. ``on_event``
    receives human-readable progress lines.

    Raises :class:`FetchError` (or :class:`DigestMismatch`) on the first pin that cannot be
    mirrored and verified. Already-present, still-verifying pins are skipped, so a second run with
    the network down succeeds from what is local.
    """
    try:
        from astro_mine.hub.client import HubClient
        from astro_mine.hub.registry import ArtifactExistsError, Blob, open_registry
    except ImportError as exc:  # pragma: no cover - exercised via the CLI's install-hint test
        raise FetchError(_INSTALL_HINT) from exc

    note = on_event if on_event is not None else (lambda _message: None)
    store_path = resolve_store_path(store)
    store_path.mkdir(parents=True, exist_ok=True)

    dest = open_registry(store_path)
    dest_client = HubClient(dest, trusted_public_key_pem=trusted_key_pem)
    source_registry: Any | None = None

    fetched: list[FetchedPin] = []
    refs = spec.content_refs()
    for index, ref in enumerate(refs, start=1):
        digest = ref.content_hash
        if _already_present(dest_client, digest):
            reference = _reference_for(dest, digest)
            note(f"[{index}/{len(refs)}] {ref.id} — already present, verified")
            fetched.append(FetchedPin(ref.id, digest, reference, mirrored=False, size_bytes=0))
            continue

        if source_registry is None:
            # Opened lazily so a fully-populated store needs no network at all (CX-LOCAL).
            source_registry = open_registry(source)
        note(f"[{index}/{len(refs)}] {ref.id} — fetching {digest[:19]}…")

        try:
            manifest = source_registry.read_manifest(digest)
            config_descriptor = manifest["config"]
            config_bytes = source_registry.pull_blob(config_descriptor["digest"])
            layers = [
                Blob(layer["mediaType"], source_registry.pull_blob(layer["digest"]))
                for layer in manifest["layers"]
            ]
        except KeyError as exc:
            raise FetchError(f"{ref.id}: malformed manifest at {digest} ({exc})") from exc

        name, version = _identity(config_bytes)
        try:
            published = dest.publish(
                name=name,
                version=version,
                kind=_artifact_kind(manifest["artifactType"]),
                config=config_bytes,
                layers=layers,
                config_media_type=config_descriptor["mediaType"],
                annotations=manifest.get("annotations"),
            )
        except ArtifactExistsError as exc:
            # The tag exists locally but the digest was absent or did not verify above — the local
            # store holds different bytes under the same name:version. Never silently prefer either.
            raise FetchError(
                f"{ref.id}: {name}:{version} already exists in {store_path} under a different "
                f"digest than the pinned {digest}; remove it or fetch into a clean store"
            ) from exc

        if published.digest != digest:
            raise DigestMismatch(
                f"{ref.id}: mirrored artifact hashes to {published.digest}, not the pinned {digest}"
            )

        _mirror_referrers(source_registry, dest, digest, Blob)

        # Verify-twice: the finished local artifact must stand on its own, attestations included.
        try:
            dest_client.verify(digest)
        except Exception as exc:
            raise FetchError(f"{ref.id}: fetched artifact failed verification: {exc}") from exc

        size = int(config_descriptor.get("size", 0)) + sum(
            int(layer.get("size", 0)) for layer in manifest["layers"]
        )
        fetched.append(
            FetchedPin(ref.id, digest, f"{name}:{version}", mirrored=True, size_bytes=size)
        )

    return tuple(fetched)


def _reference_for(registry: Any, digest: str) -> str:
    """The ``name:version`` a local artifact is indexed under; its digest if it is untagged."""
    try:
        config = json.loads(registry.read_config(digest))
        return f"{config['name']}:{config['version']}"
    except Exception:
        return digest
