"""The embedding-provider seam (RM-P1-HUB-02; hub.md §3, §11).

hub.md §3 makes the semantic backend a documented extension point. Two providers ship: the offline
feature-hashing default (tier-1 works with no model and no network — principle 7) and a **learned**
model served over an OpenAI-compatible ``/v1/embeddings`` endpoint (the hosted tier's vectors, the
ones pgvector indexes). The served provider is exercised against a real in-process HTTP endpoint, so
the request/response contract is tested rather than mocked; a provider that cannot produce a vector
raises rather than silently returning a zero vector that would rank as "no match".
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from astro_mine.hub.index import InMemoryCatalog, ingest
from astro_mine.hub.search import (
    EmbeddingError,
    EmbeddingProvider,
    HashingEmbedding,
    HttpEmbedding,
    SearchQuery,
    default_provider,
    search,
)

from .conftest import make_manifest


class _FakeEmbeddingServer:
    """A minimal OpenAI-compatible ``/v1/embeddings`` endpoint (a served model, stubbed)."""

    def __init__(self, *, vector: list[float] | None = None, malformed: bool = False) -> None:
        self.vector = vector if vector is not None else [0.0, 3.0, 4.0]
        self.malformed = malformed
        self.seen: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                server.seen.append(
                    {
                        "body": json.loads(self.rfile.read(length)),
                        "authorization": self.headers.get("Authorization"),
                    }
                )
                payload = (
                    {"nonsense": True}
                    if server.malformed
                    else {"data": [{"embedding": server.vector}]}
                )
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _FakeEmbeddingServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1/embeddings"


@pytest.fixture(autouse=True)
def _no_ambient_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("HUB_EMBEDDING_URL", "HUB_EMBEDDING_MODEL", "HUB_EMBEDDING_DIM"):
        monkeypatch.delenv(name, raising=False)
    yield


def test_hashing_provider_is_the_offline_default() -> None:
    provider = default_provider()
    assert isinstance(provider, HashingEmbedding)
    assert provider.name == "hashing-64" and provider.dim == 64
    assert isinstance(provider, EmbeddingProvider)

    vector = provider.embed("lunar excavation")
    assert len(vector) == 64
    assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-9)  # L2-normalized
    assert provider.embed("lunar excavation") == vector  # deterministic


def test_http_provider_embeds_against_a_served_model() -> None:
    with _FakeEmbeddingServer(vector=[0.0, 3.0, 4.0]) as server:
        provider = HttpEmbedding(server.url, model="e5-small", dim=3, api_key="k")
        assert provider.name == "http:e5-small" and provider.dim == 3

        vector = provider.embed("excavation policy")
        assert vector == pytest.approx((0.0, 0.6, 0.8))  # L2-normalized by the provider
        assert server.seen[0]["body"] == {"model": "e5-small", "input": "excavation policy"}
        assert server.seen[0]["authorization"] == "Bearer k"

        assert provider.embed("   ") == (0.0, 0.0, 0.0)  # empty text never hits the network
        assert len(server.seen) == 1


def test_http_provider_fails_closed_on_a_dimension_mismatch() -> None:
    with _FakeEmbeddingServer(vector=[1.0, 2.0]) as server:
        provider = HttpEmbedding(server.url, model="m", dim=384)
        with pytest.raises(EmbeddingError, match="2-d vector"):
            provider.embed("x")


def test_http_provider_fails_closed_on_a_malformed_response() -> None:
    with _FakeEmbeddingServer(malformed=True) as server:
        provider = HttpEmbedding(server.url, model="m", dim=3)
        with pytest.raises(EmbeddingError, match="malformed embedding response"):
            provider.embed("x")


def test_http_provider_fails_closed_when_unreachable() -> None:
    provider = HttpEmbedding("http://127.0.0.1:1/v1/embeddings", model="m", dim=3, timeout=1.0)
    with pytest.raises(EmbeddingError, match="failed"):
        provider.embed("x")


def test_default_provider_switches_to_the_served_model_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUB_EMBEDDING_URL", "http://embeddings.internal/v1/embeddings")
    monkeypatch.setenv("HUB_EMBEDDING_MODEL", "bge-small")
    monkeypatch.setenv("HUB_EMBEDDING_DIM", "384")

    provider = default_provider()
    assert isinstance(provider, HttpEmbedding)
    assert provider.name == "http:bge-small" and provider.dim == 384


def test_ingest_and_search_run_on_an_injected_provider() -> None:
    """The whole discovery path is provider-parametric: ingest with one, query with the same one."""
    with _FakeEmbeddingServer(vector=[1.0, 0.0, 0.0]) as server:
        provider = HttpEmbedding(server.url, model="m", dim=3)
        catalog = InMemoryCatalog()
        entry = ingest(
            catalog,
            make_manifest("excavator", "1.0.0", description="lunar excavation"),
            digest="sha256:" + "a" * 64,
            publisher="alice",
            provider=provider,
        )
        assert entry.embedding_provider == "http:m"
        assert entry.embedding == (1.0, 0.0, 0.0)

        results = search(catalog, SearchQuery(semantic="digging"), provider=provider)
        assert results[0].entry.name == "excavator"
        assert results[0].score == pytest.approx(1.0)  # cosine against the served vector
