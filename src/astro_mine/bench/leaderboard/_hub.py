"""Hub-digest submission intake (RM-P1-BENCH-10; bench.md §3, §6, §9).

The public leaderboard's flywheel turns when an external lab **publishes a policy to Hub** and it
is scored **from its digest alone** — Bench never trusts an uploaded blob, it resolves the artifact
from Hub by content hash, verifies it fail-closed, validates its Core plugin manifest against the
scenario's declared interface, and only then runs it under submit-policy-we-run (bench.md §6, §9).
This module is that intake:

- :func:`resolve_submission` — resolve a Hub reference (a ``name:version`` tag or a ``sha256:``
  digest) to its :class:`~astro_mine.core.registry.PluginManifest`, **verifying twice** (the image
  manifest, its config, and every layer) before trusting a byte (hub.md verify-twice; bench.md §9);
- :func:`validate_submission_manifest` — assert the manifest is a *policy* plugin whose declared
  ``core_interfaces`` **satisfy** the ScenarioSpec's ``core_interface`` (the observation/action
  interface contract, under the registry's SemVer rule);
- :class:`PolicyLoader` / :func:`reference_policy_loader` — materialize the resolved artifact into a
  runnable Core :class:`~astro_mine.core.policy.Policy` behind an injected seam.

Bench imports **only Core + the Hub client**: the concrete
:class:`astro_mine.hub.registry.Registry` is opened lazily by :func:`open_registry` (behind the
``[leaderboard]`` extra), while this module types the registry structurally (:class:`HubRegistry`)
so it stays import-light and never reaches into Sim or a private Hub schema (bench.md §2.2).

**Policy materialization (the injected seam).** :func:`reference_policy_loader` resolves the
manifest's declared ``entrypoint`` attribute (a ``module:attribute`` Core policy) so a submission is
runnable under the dependency-clean reference runner and in CI with no ONNX runtime. A deployment
injects an ONNX-materializing loader that binds :class:`astro_mine.core.policy.OnnxPolicy` to an
inference function over the ONNX layer it takes from
:func:`~astro_mine.bench.leaderboard.pull_verified_layer` — Bench's one door onto an artifact's
bytes, and a verified one: the layer is re-hashed against the digest the verified manifest commits
to before a byte is returned (hub.md §2.3; conventions.md §9). It is the same injection seam Bench
uses for the Sim :class:`~astro_mine.bench.baseline.EpisodeRunner` (bench.md §2.2). Bench ships no
inference runtime.

Backlog: RM-P1-BENCH-10 — astro-mine-bench#18
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from astro_mine.bench.leaderboard._eval import PolicyReferenceError, resolve_policy
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.core.compat import check_compatible
from astro_mine.core.policy import Policy
from astro_mine.core.registry import PluginKind, load_plugin_manifest
from astro_mine.core.registry.model import PluginManifest

__all__ = [
    "HubRegistry",
    "HubResolutionError",
    "ManifestInterfaceError",
    "PolicyLoader",
    "ResolvedSubmission",
    "open_registry",
    "reference_policy_loader",
    "resolve_submission",
    "submission_policy_ref",
    "validate_submission_manifest",
]


class HubResolutionError(Exception):
    """Raised when a Hub reference cannot be resolved, verified, or parsed to a manifest."""


class ManifestInterfaceError(Exception):
    """Raised when a manifest is not a policy or fails the scenario's interface contract."""


class HubRegistry(Protocol):
    """The Hub-client surface Bench consumes — met by :class:`astro_mine.hub.registry.Registry`.

    Typed structurally so Bench depends on the Hub client only where a concrete registry is opened,
    keeping the intake logic import-light and free of a private Hub schema (bench.md §2.2).
    """

    def resolve(self, reference: str) -> Any:
        """Resolve a ``name:version`` tag or a digest to its manifest descriptor (``.digest``)."""
        ...

    def verify(self, digest: str) -> None:
        """Assert the stored blob at ``digest`` hashes to it — content-addressing on read."""
        ...

    def read_manifest(self, digest: str) -> dict[str, Any]:
        """The parsed OCI image manifest at ``digest`` (``config`` + ``layers`` descriptors)."""
        ...

    def read_config(self, manifest_digest: str) -> bytes:
        """The artifact's config blob (its Core plugin manifest) given the image-manifest digest."""
        ...

    def referrers(self, subject: str, *, artifact_type: str | None = None) -> list[Any]:
        """The OCI **referrers** of ``subject`` — where a submission's attestations hang.

        The cosign signature, the SLSA provenance statement, and the SBOM are attached to the image
        manifest as referrer artifacts; supply-chain verification (:mod:`._supply_chain`) reads them
        back through this call (bench.md §9).
        """
        ...


@dataclass(frozen=True)
class ResolvedSubmission:
    """A Hub artifact resolved to its verified manifest + payload layer digests."""

    reference: str
    manifest_digest: str
    manifest: PluginManifest
    layer_digests: tuple[str, ...]


def resolve_submission(registry: HubRegistry, reference: str) -> ResolvedSubmission:
    """Resolve ``reference`` from Hub to a verified :class:`ResolvedSubmission` (bench.md §6, §9).

    Resolves the reference to its image-manifest digest, then **fail-closed verifies** the manifest,
    its config blob, and every payload layer against their content addresses before parsing the Core
    plugin manifest. Any resolution, integrity, or parse failure raises :class:`HubResolutionError`
    — a submission that cannot be authenticated by digest never runs.
    """
    try:
        descriptor = registry.resolve(reference)
    except Exception as exc:  # ArtifactNotFound and any client-specific resolution failure
        raise HubResolutionError(f"cannot resolve Hub reference {reference!r}: {exc}") from exc
    manifest_digest = str(descriptor.digest)
    try:
        # verify() re-checks the image manifest AND its config + every layer against their content
        # addresses in one call (hub.md §2.3 verify-twice) — fail-closed on any tampered blob.
        registry.verify(manifest_digest)
        image = registry.read_manifest(manifest_digest)
        layer_digests = tuple(str(layer["digest"]) for layer in image.get("layers", ()))
    except Exception as exc:
        raise HubResolutionError(f"integrity verification failed for {reference!r}: {exc}") from exc
    try:
        # The stored config blob is the **bare manifest**, not a manifest document — hub.md §2
        # principle 2, and what every publisher in this platform writes. This read used
        # `load_manifest(...).manifest`, which expects the authored envelope, so the community
        # submission path could not accept a single artifact the platform publishes: it answered
        # 404 content_not_found with a schema error naming every real field as unexpected
        # (astro-mine-platform#14). `load_plugin_manifest` reads the stored form and keeps the
        # checks the envelope loader was giving us — the gated-capability-tag gate above all,
        # which is not optional on the one path where a third party's manifest arrives.
        manifest = load_plugin_manifest(registry.read_config(manifest_digest))
    except Exception as exc:
        raise HubResolutionError(f"invalid plugin manifest for {reference!r}: {exc}") from exc
    return ResolvedSubmission(reference, manifest_digest, manifest, layer_digests)


def validate_submission_manifest(resolved: ResolvedSubmission, spec: ScenarioSpec) -> None:
    """Assert the submission is a policy whose interface satisfies the scenario (bench.md §6).

    The manifest must declare :attr:`PluginKind.POLICY` and its ``core_interfaces`` must **satisfy**
    every interface the ScenarioSpec pins (the observation/action interface contract), under the
    same SemVer rule the Core registry applies at load. Raises :class:`ManifestInterfaceError`
    otherwise — an interface-incompatible submission is rejected before it runs.
    """
    manifest = resolved.manifest
    if manifest.kind is not PluginKind.POLICY:
        raise ManifestInterfaceError(
            f"submission {resolved.reference!r} must be a policy plugin, got kind={manifest.kind}"
        )
    for name, required in spec.core_interface.items():
        provided = manifest.core_interfaces.get(name)
        if provided is None or not check_compatible(required, provided):
            raise ManifestInterfaceError(
                f"submission {resolved.reference!r} manifest interfaces {manifest.core_interfaces} "
                f"do not satisfy scenario {spec.scenario_id!r} interface {name}>={required} "
                f"(declared {provided!r})"
            )


def submission_policy_ref(resolved: ResolvedSubmission) -> str:
    """The submission's ``module:attribute`` policy reference — **read, not imported** (bench#30).

    The evaluator needs to know *what* to run, not to *load* it: this returns the declared
    ``entrypoint`` as a plain string, which the service hands to the sandboxed eval worker. The
    import happens inside the sandbox, in the worker's process, under the no-egress envelope — never
    in the leaderboard's, which is what :func:`reference_policy_loader` would do (bench.md §9).

    Raises :class:`HubResolutionError` if the manifest declares no entrypoint; a deployment that
    accepts raw ONNX submissions materializes them inside the *worker image* instead.
    """
    entrypoint = resolved.manifest.attributes.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise HubResolutionError(
            f"submission {resolved.reference!r} manifest declares no 'entrypoint' attribute; "
            "ship an ONNX-materializing entrypoint in the sandboxed evaluation-runner image to run "
            "a raw ONNX submission"
        )
    return entrypoint


class PolicyLoader(Protocol):
    """Materialize a resolved Hub submission into a runnable Core :class:`Policy` (the seam).

    A loader that needs the submission's *bytes* — an ONNX graph rather than an importable
    entrypoint — takes them from :func:`~astro_mine.bench.leaderboard.pull_verified_layer`, the one
    route Bench's registry seam offers: the layer is re-hashed against the digest the verified
    manifest commits to, and a digest that manifest does not commit to is refused (hub.md §2.3;
    conventions.md §9). Bytes no manifest vouched for are not a policy.

    .. warning::
       This runs **in the calling process**. Since bench#30 the leaderboard no longer calls it:
       materializing a community submission inside the evaluator is exactly the trust violation
       bench.md §9 forbids. A loader is now something the *sandboxed worker* applies, inside the
       evaluation-runner image; the service only ever handles the reference string
       (:func:`submission_policy_ref`).
    """

    def __call__(self, resolved: ResolvedSubmission, registry: HubRegistry) -> Policy:
        """Build the policy — only ever inside the sandbox, never in the leaderboard evaluator."""
        ...


def reference_policy_loader(resolved: ResolvedSubmission, registry: HubRegistry) -> Policy:
    """Materialize the manifest's ``entrypoint`` into a live Policy — **inside the sandbox only**.

    The dependency-clean :class:`PolicyLoader`: it resolves the manifest's ``module:attribute``
    entrypoint (via :func:`~astro_mine.bench.leaderboard.resolve_policy`) so a submission that ships
    a Python entrypoint runs under the reference runner with no ONNX runtime.

    .. warning::
       Calling this **imports community code into the calling process**. It belongs in the sandboxed
       evaluation runner, not in the leaderboard service — which since bench#30 handles only the
       reference string (:func:`submission_policy_ref`) and never the object.

    Raises :class:`HubResolutionError` if the manifest declares no entrypoint, or it fails to load.
    """
    entrypoint = submission_policy_ref(resolved)
    try:
        return resolve_policy(entrypoint)
    except PolicyReferenceError as exc:
        raise HubResolutionError(
            f"submission {resolved.reference!r} entrypoint {entrypoint!r} did not load: {exc}"
        ) from exc


def open_registry(path: str | Path) -> HubRegistry:
    """Open the content-addressed Hub registry at ``path`` (requires the ``[leaderboard]`` extra).

    Thin lazy wrapper over :class:`astro_mine.hub.registry.Registry` so the base package imports
    without the Hub client; ``path`` is the workspace tier-1 registry (the ``files/hub-registry``
    convention).
    """
    from astro_mine.hub.registry import Registry

    return Registry(path)
