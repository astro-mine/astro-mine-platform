"""An in-process **OCI Distribution Spec** registry — the offline test double for RM-P1-HUB-06.

:class:`RemoteRegistry` speaks HTTP, so testing it against a mock object would test nothing: the
whole point is spec conformance on the wire. This is a real (if minimal) registry — a threaded HTTP
server implementing ``/v2/`` ping, manifest PUT/GET/HEAD, the blob upload session (POST → PUT), blob
GET/HEAD, tag listing, ``_catalog``, and the **Referrers API** — so the offline suite exercises the
same code path CI exercises against a real Zot, with no service to run.

It is deliberately *configurable in its conformance* so the client's degradation paths are testable:
``supports_referrers=False`` forces the spec's fallback-tag behaviour, ``supports_catalog=False``
mimics ghcr (no ``_catalog``), and ``require_auth=True`` forces the full
``WWW-Authenticate: Bearer`` token handshake. :meth:`FakeRegistry.tamper` corrupts a stored blob so
the client's fail-closed integrity check can be proven, not asserted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

_MEDIA_INDEX = "application/vnd.oci.image.index.v1+json"
_MEDIA_MANIFEST = "application/vnd.oci.image.manifest.v1+json"


@dataclass
class _Repo:
    blobs: dict[str, bytes] = field(default_factory=dict)
    manifests: dict[str, bytes] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)  # tag → manifest digest


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class FakeRegistry:
    """A minimal but spec-faithful OCI registry served over real HTTP on localhost."""

    def __init__(
        self,
        *,
        require_auth: bool = False,
        supports_referrers: bool = True,
        supports_catalog: bool = True,
        username: str = "user",
        password: str = "pass",
    ) -> None:
        self.repos: dict[str, _Repo] = {}
        self.uploads: dict[str, str] = {}  # upload id → repository
        self.require_auth = require_auth
        self.supports_referrers = supports_referrers
        self.supports_catalog = supports_catalog
        self.username = username
        self.password = password
        self.token = "fake-bearer-token"
        self.token_requests = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle -------------------------------------------------------------------------------

    def __enter__(self) -> FakeRegistry:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def location(self) -> str:
        """The plain-HTTP location a :class:`RemoteRegistry` is opened against."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # -- test hooks ------------------------------------------------------------------------------

    def repo(self, name: str) -> _Repo:
        return self.repos.setdefault(name, _Repo())

    def tamper(self, repository: str, digest: str, data: bytes) -> None:
        """Serve ``data`` at ``digest`` — a registry lying about a content address."""
        self.repo(repository).blobs[digest] = data


def _make_handler(registry: FakeRegistry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep the test output clean
            pass

        # -- helpers ---------------------------------------------------------------------------

        def _send(
            self, status: int, body: bytes = b"", headers: dict[str, str] | None = None
        ) -> None:
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _authorized(self) -> bool:
            if not registry.require_auth:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {registry.token}"

        def _challenge(self) -> None:
            realm = f"{registry.location}/token"
            self._send(
                401,
                b'{"errors":[{"code":"UNAUTHORIZED"}]}',
                {"WWW-Authenticate": f'Bearer realm="{realm}",service="fake",scope="repository:*"'},
            )

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _token(self) -> None:
            """The token endpoint: exchange Basic credentials for a bearer token."""
            registry.token_requests += 1
            expected = base64.b64encode(
                f"{registry.username}:{registry.password}".encode()
            ).decode()
            if self.headers.get("Authorization") != f"Basic {expected}":
                self._send(401, b'{"errors":[{"code":"UNAUTHORIZED"}]}')
                return
            self._send(200, json.dumps({"token": registry.token}).encode())

        def _referrers(self, repository: str, subject: str) -> None:
            if not registry.supports_referrers:
                self._send(404, b'{"errors":[{"code":"UNSUPPORTED"}]}')
                return
            repo = registry.repo(repository)
            manifests = []
            for digest, raw in repo.manifests.items():
                manifest = json.loads(raw)
                if (manifest.get("subject") or {}).get("digest") != subject:
                    continue
                manifests.append(
                    {
                        "mediaType": _MEDIA_MANIFEST,
                        "digest": digest,
                        "size": len(raw),
                        "artifactType": manifest.get("artifactType"),
                    }
                )
            index = {"schemaVersion": 2, "mediaType": _MEDIA_INDEX, "manifests": manifests}
            self._send(200, json.dumps(index).encode(), {"Content-Type": _MEDIA_INDEX})

        # -- verbs ------------------------------------------------------------------------------

        def do_GET(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if path == "/token":
                self._token()
                return
            if not self._authorized():
                self._challenge()
                return
            if path == "/v2/":
                self._send(200, b"{}")
                return
            if path == "/v2/_catalog":
                if not registry.supports_catalog:
                    self._send(404, b'{"errors":[{"code":"UNSUPPORTED"}]}')
                    return
                body = json.dumps({"repositories": sorted(registry.repos)}).encode()
                self._send(200, body)
                return

            parts = path.strip("/").split("/")
            if len(parts) < 4 or parts[0] != "v2":
                self._send(404, b'{"errors":[{"code":"NOT_FOUND"}]}')
                return
            kind, reference = parts[-2], parts[-1]
            repository = "/".join(parts[1:-2])
            repo = registry.repo(repository)

            if kind == "manifests":
                digest = repo.tags.get(reference, reference)
                raw = repo.manifests.get(digest)
                if raw is None:
                    self._send(404, b'{"errors":[{"code":"MANIFEST_UNKNOWN"}]}')
                    return
                self._send(
                    200,
                    raw,
                    {"Content-Type": _MEDIA_MANIFEST, "Docker-Content-Digest": _digest(raw)},
                )
            elif kind == "blobs":
                data = repo.blobs.get(reference)
                if data is None:
                    self._send(404, b'{"errors":[{"code":"BLOB_UNKNOWN"}]}')
                    return
                self._send(200, data, {"Content-Type": "application/octet-stream"})
            elif kind == "referrers":
                self._referrers(repository, reference)
            elif kind == "tags" and reference == "list":
                body = json.dumps({"name": repository, "tags": sorted(repo.tags)}).encode()
                self._send(200, body)
            else:
                self._send(404, b'{"errors":[{"code":"NOT_FOUND"}]}')

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_POST(self) -> None:
            if not self._authorized():
                self._challenge()
                return
            parts = self.path.strip("/").split("/")
            if parts[-2:] != ["blobs", "uploads"]:
                self._send(404, b'{"errors":[{"code":"NOT_FOUND"}]}')
                return
            repository = "/".join(parts[1:-2])
            upload = f"upload-{len(registry.uploads)}"
            registry.uploads[upload] = repository
            self._send(
                202, b"", {"Location": f"/v2/{repository}/blobs/uploads/{upload}", "Range": "0-0"}
            )

        def do_PUT(self) -> None:
            if not self._authorized():
                self._challenge()
                return
            parsed = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            parts = parsed.path.strip("/").split("/")
            body = self._body()

            if "uploads" in parts:  # PUT /v2/<repo>/blobs/uploads/<id>?digest=...
                upload = parts[-1]
                repository = registry.uploads.get(upload)
                digest = (query.get("digest") or [""])[0]
                if repository is None or not digest:
                    self._send(400, b'{"errors":[{"code":"BLOB_UPLOAD_INVALID"}]}')
                    return
                if _digest(body) != digest:
                    self._send(400, b'{"errors":[{"code":"DIGEST_INVALID"}]}')
                    return
                registry.repo(repository).blobs[digest] = body
                self._send(
                    201,
                    b"",
                    {
                        "Location": f"/v2/{repository}/blobs/{digest}",
                        "Docker-Content-Digest": digest,
                    },
                )
                return

            if parts[-2] == "manifests":  # PUT /v2/<repo>/manifests/<tag|digest>
                reference = parts[-1]
                repository = "/".join(parts[1:-2])
                repo = registry.repo(repository)
                digest = _digest(body)
                repo.manifests[digest] = body
                if not reference.startswith("sha256:"):
                    repo.tags[reference] = digest
                headers = {"Docker-Content-Digest": digest}
                subject = (json.loads(body).get("subject") or {}).get("digest")
                if subject and registry.supports_referrers:
                    headers["OCI-Subject"] = subject
                self._send(201, b"", headers)
                return

            self._send(404, b'{"errors":[{"code":"NOT_FOUND"}]}')

    return Handler
