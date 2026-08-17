# SPDX-License-Identifier: Apache-2.0
"""The OCI **Distribution-Spec** transport — pull/publish against any registry (RM-P1-HUB-06).

hub.md §7 requires the client to "resolve/verify/pull against **any** OCI registry (including
``ghcr.io`` or a private Harbor/Zot) so a researcher needs no hosted Hub", and §2 principle 6
forbids a bespoke protocol. :class:`RemoteRegistry` is that transport: a spec-faithful HTTP client
for the ``/v2/…`` endpoints (manifests, blobs, uploads, tags, **referrers**), satisfying the same
:class:`~astro_mine.hub.registry._protocol.RegistryClient` contract as the local OCI-layout
:class:`~astro_mine.hub.registry._store.Registry` — so :mod:`astro_mine.hub.supply_chain`, the
client, and the CLI are written once and work against either.

**Hand-rolled on ``urllib``, deliberately.** The tier-1 path (local layout + client) MUST work fully
offline (hub.md principle 7), so the base package stays dependency-clean — a remote pull adds no
third-party HTTP stack, exactly as :mod:`astro_mine.hub.registry._oci` hand-rolls the image-layout
rather than depending on an ORAS runtime.

**Verify-twice survives the wire (hub.md §2.3, §9).** :meth:`resolve` hash-checks the manifest bytes
it fetched against the digest the registry claims (a lying ``Docker-Content-Digest`` is caught, not
trusted), and :meth:`pull_blob` re-hashes **every** blob before returning it, so a compromised or
man-in-the-middled registry cannot make a consumer accept tampered bytes — it raises
:class:`~astro_mine.hub.registry._oci.IntegrityError` and **fails closed**. Nothing here returns
unverified bytes.

**Auth** is resolved by :mod:`astro_mine.hub.registry._auth` from the standard Docker credential
sources, and the token (``WWW-Authenticate: Bearer``) handshake is performed transparently;
``Authorization`` is **dropped on a cross-host redirect** (blob egress commonly 307s to object
storage) so credentials never leak to a CDN/S3 origin.

**Repository layout.** An artifact ``name:version`` maps to the conventional OCI coordinates
``<prefix>/<name>:<version>`` — i.e. ``ghcr.io/astro-mine/pol:1.0.0`` — one repository per artifact
name. Blobs in the Distribution Spec are repository-scoped, so a *bare* ``sha256:…`` reference is
only resolvable once the digest's repository is known (from a prior resolve/publish in this client);
use the unambiguous ``name@sha256:…`` form to address an artifact by digest cold.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from astro_mine.hub._content import canonical_json, content_hash
from astro_mine.hub.registry._auth import Credentials, credentials_for
from astro_mine.hub.registry._oci import (
    EMPTY_CONFIG,
    MEDIA_CORE_MANIFEST,
    MEDIA_EMPTY,
    MEDIA_INDEX,
    MEDIA_MANIFEST,
    Blob,
    Descriptor,
    IntegrityError,
    artifact_media_type,
    canonical_manifest,
)
from astro_mine.hub.registry._store import (
    ArtifactExistsError,
    ArtifactNotFound,
    PublishedArtifact,
    _parse_reference,
)

__all__ = ["RegistryHttpError", "RemoteRegistry", "is_remote"]

_DEFAULT_TIMEOUT_S = 60.0
_MANIFEST_ACCEPT = f"{MEDIA_MANIFEST}, {MEDIA_INDEX}"
_CHALLENGE = re.compile(r'(\w+)="([^"]*)"')


class RegistryHttpError(Exception):
    """A remote registry returned an unexpected HTTP status (the body is included when present)."""

    def __init__(self, method: str, url: str, status: int, body: bytes = b"") -> None:
        detail = body.decode("utf-8", "replace").strip()
        super().__init__(f"{method} {url} → HTTP {status}{f': {detail}' if detail else ''}")
        self.status = status


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects but **strip ``Authorization`` when the host changes** (no credential leak).

    Blob egress typically 307s to object storage / a CDN; forwarding the registry's bearer token
    there would leak the credential (and most S3 origins reject a second auth mechanism outright).
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            new is not None
            and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc
        ):
            new.remove_header("Authorization")
        return new


class RemoteRegistry:
    """A content-addressed OCI registry reached over the **OCI Distribution Spec** (HTTP).

    ``location`` is the registry root, optionally with a repository prefix and scheme —
    ``ghcr.io/astro-mine``, ``https://registry.example.org/commons``, ``http://localhost:5000``
    (plain HTTP is used when the scheme says so, for a local Zot/registry in dev and CI).
    Credentials come from the standard Docker sources unless passed explicitly.
    """

    def __init__(
        self,
        location: str,
        *,
        credentials: Credentials | None = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        scheme, host, prefix = _split_location(location)
        self.scheme = scheme
        self.host = host
        self.prefix = prefix
        self.timeout = timeout
        self._credentials = credentials if credentials is not None else credentials_for(host)
        self._tokens: dict[str, str] = {}  # scope → bearer token (cached per repository scope)
        self._repo_by_digest: dict[str, str] = {}  # digest → repository (blobs are repo-scoped)
        self._opener = urllib.request.build_opener(_Redirects)

    # -- HTTP + auth ---------------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.scheme}://{self.host}{path}"

    def _repository(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _authorize(self, request: urllib.request.Request, scope: str) -> None:
        if token := self._tokens.get(scope):
            request.add_header("Authorization", f"Bearer {token}")
        elif self._credentials is not None:
            request.add_header("Authorization", f"Basic {self._credentials.basic}")

    def _fetch_token(self, challenge: str, scope: str) -> bool:
        """Complete a ``WWW-Authenticate: Bearer`` handshake; ``True`` if a token was obtained."""
        scheme, _, params = challenge.partition(" ")
        if scheme.lower() != "bearer":
            return False
        fields = dict(_CHALLENGE.findall(params))
        realm = fields.get("realm")
        if not realm:
            return False
        query = {k: v for k, v in fields.items() if k in ("service",) and v}
        query["scope"] = fields.get("scope") or scope
        request = urllib.request.Request(f"{realm}?{urllib.parse.urlencode(query)}")
        if self._credentials is not None:
            request.add_header("Authorization", f"Basic {self._credentials.basic}")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, ValueError):
            return False
        token = payload.get("token") or payload.get("access_token")
        if not isinstance(token, str) or not token:
            return False
        self._tokens[scope] = token
        return True

    def _call(
        self,
        method: str,
        url: str,
        *,
        scope: str,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        expected: Sequence[int] = (200,),
    ) -> tuple[int, dict[str, str], bytes]:
        """One authenticated request, retried once through a bearer-token handshake on 401."""
        for attempt in (0, 1):
            request = urllib.request.Request(url, data=body, method=method)
            for key, value in (headers or {}).items():
                request.add_header(key, value)
            self._authorize(request, scope)
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    payload = response.read()
                    return int(response.status), dict(response.headers), payload
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                if exc.code == 401 and attempt == 0:
                    challenge = exc.headers.get("WWW-Authenticate", "")
                    if self._fetch_token(challenge, scope):
                        continue
                if exc.code in expected:
                    return int(exc.code), dict(exc.headers), payload
                raise RegistryHttpError(method, url, int(exc.code), payload) from exc
            except urllib.error.URLError as exc:
                raise RegistryHttpError(method, url, 0, str(exc.reason).encode()) from exc
        raise RegistryHttpError(method, url, 401)  # pragma: no cover - loop always returns/raises

    def _pull_scope(self, repository: str) -> str:
        return f"repository:{repository}:pull"

    def _push_scope(self, repository: str) -> str:
        return f"repository:{repository}:pull,push"

    # -- repository bookkeeping ----------------------------------------------------------------

    def _remember(self, repository: str, *digests: str) -> None:
        for digest in digests:
            self._repo_by_digest[digest] = repository

    def _repo_for_digest(self, digest: str) -> str:
        try:
            return self._repo_by_digest[digest]
        except KeyError:
            raise ArtifactNotFound(
                f"{digest} is not associated with a repository on {self.host}; blobs are "
                f"repository-scoped in the OCI Distribution Spec — resolve the artifact first, or "
                f"address it as 'name@{digest}'"
            ) from None

    # -- publish / resolve / pull ---------------------------------------------------------------

    def _blob_exists(self, repository: str, digest: str) -> bool:
        status, _, _ = self._call(
            "HEAD",
            self._url(f"/v2/{repository}/blobs/{digest}"),
            scope=self._pull_scope(repository),
            expected=(200, 404),
        )
        return status == 200

    def _put_blob(self, repository: str, blob: Blob) -> Descriptor:
        """Upload ``blob`` (POST an upload session, then a monolithic PUT); idempotent."""
        if not self._blob_exists(repository, blob.digest):
            _, headers, _ = self._call(
                "POST",
                self._url(f"/v2/{repository}/blobs/uploads/"),
                scope=self._push_scope(repository),
                headers={"Content-Length": "0"},
                expected=(202, 201),
            )
            location = headers.get("Location")
            if not location:
                raise RegistryHttpError("POST", f"/v2/{repository}/blobs/uploads/", 202)
            upload = urllib.parse.urljoin(self._url(f"/v2/{repository}/blobs/uploads/"), location)
            joiner = "&" if urllib.parse.urlsplit(upload).query else "?"
            self._call(
                "PUT",
                f"{upload}{joiner}digest={urllib.parse.quote(blob.digest)}",
                scope=self._push_scope(repository),
                headers={"Content-Type": "application/octet-stream"},
                body=blob.data,
                expected=(201,),
            )
        self._remember(repository, blob.digest)
        return blob.descriptor()

    def _put_manifest(self, repository: str, reference: str, blob: Blob) -> dict[str, str]:
        _, headers, _ = self._call(
            "PUT",
            self._url(f"/v2/{repository}/manifests/{reference}"),
            scope=self._push_scope(repository),
            headers={"Content-Type": MEDIA_MANIFEST},
            body=blob.data,
            expected=(201,),
        )
        self._remember(repository, blob.digest)
        return headers

    def _get_manifest(self, repository: str, reference: str) -> bytes:
        _, _, body = self._call(
            "GET",
            self._url(f"/v2/{repository}/manifests/{reference}"),
            scope=self._pull_scope(repository),
            headers={"Accept": _MANIFEST_ACCEPT},
        )
        return body

    @staticmethod
    def _checked(digest: str, body: bytes, what: str) -> bytes:
        """Return ``body`` only if it hashes to ``digest`` — otherwise fail closed."""
        actual = content_hash(body)
        if actual != digest:
            raise IntegrityError(
                f"{what} {digest} content-address mismatch (served bytes hash {actual})"
            )
        return body

    def publish(
        self,
        *,
        name: str,
        version: str,
        kind: str,
        config: bytes | Mapping[str, Any],
        layers: Sequence[Blob] = (),
        config_media_type: str = MEDIA_CORE_MANIFEST,
        annotations: Mapping[str, str] | None = None,
    ) -> PublishedArtifact:
        """Push a typed artifact to ``<prefix>/<name>:<version>`` — immutable, content-addressed.

        Raises :class:`~astro_mine.hub.registry.ArtifactExistsError` if the tag already exists (a
        re-publish to an existing version is rejected — publish a new version; hub.md §2.1).
        """
        repository = self._repository(name)
        status, _, _ = self._call(
            "HEAD",
            self._url(f"/v2/{repository}/manifests/{version}"),
            scope=self._pull_scope(repository),
            headers={"Accept": _MANIFEST_ACCEPT},
            expected=(200, 404),
        )
        if status == 200:
            raise ArtifactExistsError(
                f"{name}:{version} already published on {self.host}; digests are immutable — "
                f"publish a new version"
            )

        config_bytes = config if isinstance(config, bytes) else canonical_json(dict(config))
        config_desc = self._put_blob(repository, Blob(config_media_type, config_bytes))
        layer_descs = [self._put_blob(repository, blob) for blob in layers]
        image_manifest: dict[str, Any] = {
            "schemaVersion": 2,
            "mediaType": MEDIA_MANIFEST,
            "artifactType": artifact_media_type(kind),
            "config": config_desc.as_dict(),
            "layers": [desc.as_dict() for desc in layer_descs],
        }
        if annotations:
            image_manifest["annotations"] = dict(annotations)
        manifest = canonical_manifest(image_manifest)
        self._put_manifest(repository, version, manifest)
        return PublishedArtifact(name, version, f"{name}:{version}", manifest.digest)

    def resolve(self, reference: str) -> Descriptor:
        """Resolve a ``name:version`` tag (or a digest form) to its manifest descriptor.

        The fetched manifest bytes are **hash-checked against the digest the registry claims** — the
        first half of verify-twice; a registry that serves bytes not matching its own
        ``Docker-Content-Digest`` fails closed here rather than being trusted.
        """
        name, version, digest = _parse_reference(reference)
        if digest is not None:
            repository = self._repository(name) if name else self._repo_for_digest(digest)
            target = digest
        else:
            repository = self._repository(name)
            target = version

        try:
            body = self._get_manifest(repository, target)
        except RegistryHttpError as exc:
            if exc.status in (404, 400):
                raise ArtifactNotFound(reference) from exc
            raise

        if digest is not None:
            self._checked(digest, body, "manifest")
        actual = content_hash(body)
        self._remember(repository, actual)
        return Descriptor(MEDIA_MANIFEST, actual, len(body))

    def read_manifest(self, digest: str) -> dict[str, Any]:
        """The parsed image manifest at ``digest`` — bytes re-hashed before they are parsed."""
        repository = self._repo_for_digest(digest)
        body = self._checked(digest, self._get_manifest(repository, digest), "manifest")
        manifest: dict[str, Any] = json.loads(body)
        return manifest

    def pull_blob(self, digest: str) -> bytes:
        """The bytes at ``digest`` — **re-hashed before they are returned** (never unverified).

        A registry that serves bytes whose hash is not ``digest`` raises :class:`IntegrityError`;
        no caller ever receives unverified content from the wire (hub.md §2.3, "verify first").
        """
        repository = self._repo_for_digest(digest)
        try:
            _, _, body = self._call(
                "GET",
                self._url(f"/v2/{repository}/blobs/{digest}"),
                scope=self._pull_scope(repository),
            )
        except RegistryHttpError as exc:
            if exc.status == 404:
                raise KeyError(digest) from exc
            raise
        return self._checked(digest, body, "blob")

    def read_config(self, manifest_digest: str) -> bytes:
        """The artifact's config blob (its Core plugin manifest), integrity-checked."""
        image = self.read_manifest(manifest_digest)
        self._remember(self._repo_for_digest(manifest_digest), image["config"]["digest"])
        return self.pull_blob(image["config"]["digest"])

    # -- referrers (attestations) ---------------------------------------------------------------

    def _fallback_tag(self, digest: str) -> str:
        """The Referrers **fallback tag** (``sha256-<hex>``) for registries without the API."""
        return digest.replace(":", "-")

    def attach(
        self,
        *,
        subject: str,
        artifact_type: str,
        blob: Blob,
        annotations: Mapping[str, str] | None = None,
    ) -> Descriptor:
        """Attach an attestation to ``subject`` via the OCI **Referrers** model.

        Pushes a referrer manifest whose ``subject`` is the artifact. If the registry does not
        implement the Referrers API (no ``OCI-Subject`` response header), the spec's **fallback
        tag** (``sha256-<hex>``) index is created/extended so :meth:`referrers` still finds it.
        """
        repository = self._repo_for_digest(subject)
        subject_desc = self.resolve(subject)
        empty = self._put_blob(repository, Blob(MEDIA_EMPTY, EMPTY_CONFIG))
        blob_desc = self._put_blob(repository, blob)
        referrer: dict[str, Any] = {
            "schemaVersion": 2,
            "mediaType": MEDIA_MANIFEST,
            "artifactType": artifact_type,
            "config": empty.as_dict(),
            "layers": [blob_desc.as_dict()],
            "subject": {
                "mediaType": MEDIA_MANIFEST,
                "digest": subject,
                "size": subject_desc.size,
            },
        }
        if annotations:
            referrer["annotations"] = dict(annotations)
        manifest = canonical_manifest(referrer)
        headers = self._put_manifest(repository, manifest.digest, manifest)
        descriptor = Descriptor(
            MEDIA_MANIFEST, manifest.digest, manifest.size, artifact_type=artifact_type
        )
        if "OCI-Subject" not in headers:
            self._extend_fallback_index(repository, subject, descriptor)
        return descriptor

    def _extend_fallback_index(self, repository: str, subject: str, desc: Descriptor) -> None:
        tag = self._fallback_tag(subject)
        try:
            existing = json.loads(self._get_manifest(repository, tag))
            manifests = list(existing.get("manifests", []))
        except (RegistryHttpError, ValueError):
            manifests = []
        if all(entry.get("digest") != desc.digest for entry in manifests):
            manifests.append(desc.as_dict())
        body = canonical_json(
            {"schemaVersion": 2, "mediaType": MEDIA_INDEX, "manifests": manifests}
        )
        index = Blob(MEDIA_INDEX, body)
        self._call(
            "PUT",
            self._url(f"/v2/{repository}/manifests/{tag}"),
            scope=self._push_scope(repository),
            headers={"Content-Type": MEDIA_INDEX},
            body=index.data,
            expected=(201,),
        )

    def referrers(self, subject: str, *, artifact_type: str | None = None) -> list[Descriptor]:
        """Attestation manifests whose subject is ``subject`` (Referrers API, fallback tag if not).

        The ``artifactType`` filter is applied client-side as well as on the wire: a registry MAY
        ignore the query parameter, and a filter that silently does nothing would weaken the
        verify step that depends on it.
        """
        repository = self._repo_for_digest(subject)
        path = f"/v2/{repository}/referrers/{subject}"
        if artifact_type:
            path += f"?artifactType={urllib.parse.quote(artifact_type)}"
        try:
            _, _, body = self._call(
                "GET", self._url(path), scope=self._pull_scope(repository), expected=(200,)
            )
            index = json.loads(body)
        except (RegistryHttpError, ValueError):
            try:  # the spec's fallback for registries without the Referrers API
                index = json.loads(self._get_manifest(repository, self._fallback_tag(subject)))
            except (RegistryHttpError, ValueError):
                return []

        found: list[Descriptor] = []
        for entry in index.get("manifests", []):
            descriptor = Descriptor.from_dict(entry)
            if artifact_type is not None and descriptor.artifact_type != artifact_type:
                continue
            self._remember(repository, descriptor.digest)
            found.append(descriptor)
        return found

    # -- catalog / listing ----------------------------------------------------------------------

    def _repositories(self) -> list[str]:
        try:
            _, _, body = self._call("GET", self._url("/v2/_catalog"), scope="registry:catalog:*")
            repositories = json.loads(body).get("repositories", [])
        except (RegistryHttpError, ValueError):
            return []  # not every registry exposes _catalog (ghcr does not) — degrade, don't fail
        prefix = f"{self.prefix}/" if self.prefix else ""
        return [
            repo[len(prefix) :]
            for repo in repositories
            if isinstance(repo, str) and repo.startswith(prefix) and "/" not in repo[len(prefix) :]
        ]

    def versions(self, name: str) -> list[str]:
        """The published versions (tags) of ``name``, sorted; referrer fallback tags excluded."""
        repository = self._repository(name)
        try:
            _, _, body = self._call(
                "GET",
                self._url(f"/v2/{repository}/tags/list"),
                scope=self._pull_scope(repository),
            )
            tags = json.loads(body).get("tags") or []
        except (RegistryHttpError, ValueError):
            return []
        return sorted(tag for tag in tags if isinstance(tag, str) and not tag.startswith("sha256-"))

    def references(self) -> list[str]:
        """Every published ``name:version``, sorted (empty if the registry has no catalog API)."""
        return sorted(
            f"{name}:{version}" for name in self._repositories() for version in self.versions(name)
        )

    # -- integrity ------------------------------------------------------------------------------

    def verify(self, digest: str) -> None:
        """Re-check the manifest and all its blobs against their content addresses (hub.md §2.3).

        Every fetch on this transport hash-checks its own bytes, so this walks the artifact's full
        blob set — a tampered config or payload layer raises :class:`IntegrityError`.
        """
        manifest = self.read_manifest(digest)
        repository = self._repo_for_digest(digest)
        for descriptor in [manifest["config"], *manifest["layers"]]:
            self._remember(repository, descriptor["digest"])
            self.pull_blob(descriptor["digest"])  # re-hashes; raises IntegrityError on a mismatch


def _split_location(location: str) -> tuple[str, str, str]:
    """Split ``[scheme://]host[:port][/prefix]`` into ``(scheme, host, prefix)``.

    Plain HTTP is used only when the caller says so explicitly (``http://localhost:5000`` — a dev/CI
    Zot); otherwise HTTPS, so a typo never silently downgrades a real pull to cleartext. ``oci://``
    is accepted (the scheme ORAS prints) and means HTTPS.
    """
    text = location.strip()
    if text.startswith(("http://", "https://", "oci://")):
        parts = urllib.parse.urlsplit(text)
        scheme = "http" if parts.scheme == "http" else "https"
        host, path = parts.netloc, parts.path
    else:
        scheme = "https"
        host, _, path = text.partition("/")
        path = f"/{path}" if path else ""
    if not host:
        raise ValueError(f"malformed registry location {location!r}; expected 'host[:port]/prefix'")
    return scheme, host, path.strip("/")


def is_remote(location: str) -> bool:
    """Whether ``location`` names a remote registry (a URL / ``host[:port]/prefix``), not a path.

    A local OCI-layout directory is a filesystem path; anything with a URL scheme, or whose first
    segment looks like a host (``ghcr.io``, ``localhost:5000``), is a registry.
    """
    text = location.strip()
    if text.startswith(("http://", "https://", "oci://")):
        return True
    if text.startswith((".", "/", "~")):
        return False
    head = text.split("/", 1)[0]
    return "." in head or ":" in head
