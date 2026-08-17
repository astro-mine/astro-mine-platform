# SPDX-License-Identifier: Apache-2.0
"""The embedding-provider seam — swappable semantic backends (RM-P1-HUB-02; hub.md §3, §11).

hub.md §3 names the search backend an **extension point** ("the semantic/full-text backend is
swappable behind the ``search/`` interface"), and §11 recommends a *learned* embedding + pgvector at
scale while §8 keeps the whole thing honest: "search may fall back to faceted/keyword when the
embedding index is unavailable" (principle 9). :class:`EmbeddingProvider` is that seam. Two
implementations ship:

- :class:`HashingEmbedding` — the **default**: a deterministic feature-hashing bag-of-tokens vector
  (:mod:`astro_mine.hub._embed`). No model, no network, no dependency — so tier-1 semantic search
  works **fully offline** (hub.md principle 7; conventions.md §7 tier 1). It is a real vector-space
  model, just a lexical one.
- :class:`HttpEmbedding` — a **learned** model served over an OpenAI-compatible ``/v1/embeddings``
  endpoint (Text-Embeddings-Inference, Ollama, vLLM, a cloud API — anything that speaks the
  de-facto standard shape). This is the hosted tier's provider: the vectors that make "find a policy
  like X" actually useful, and the ones the pgvector column indexes (hub.md §8, §11). It speaks HTTP
  over ``urllib`` — no new dependency lands in the offline path to get it.

A provider is identified by :attr:`~EmbeddingProvider.name` and :attr:`~EmbeddingProvider.dim`.
Vectors from different providers are **not comparable**, so the name is what a catalog records to
know whether a stored vector was produced by the provider now being queried with; the pgvector
column is created at the provider's dimensionality.

Any other backend (``sentence-transformers`` in-process, OpenSearch's vectorizer, …) is a class with
these three members — no Hub change required.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from astro_mine.hub._embed import EMBED_DIM, embed

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "HashingEmbedding",
    "HttpEmbedding",
    "default_provider",
]

_HTTP_TIMEOUT_S = 30.0


class EmbeddingError(Exception):
    """An embedding provider could not produce a vector (network, auth, or a malformed response)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text → vector. The seam :mod:`astro_mine.hub.search` and the catalog are written against."""

    @property
    def name(self) -> str:
        """The provider's identity — recorded with a vector; vectors of different names differ."""
        ...

    @property
    def dim(self) -> int:
        """The vector dimensionality (the pgvector column is created at this width)."""
        ...

    def embed(self, text: str) -> tuple[float, ...]:
        """The L2-normalized embedding of ``text`` (a zero vector for empty text)."""
        ...


class HashingEmbedding:
    """The offline default: a deterministic feature-hashing vector (no model, no network)."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> tuple[float, ...]:
        return embed(text, dim=self._dim)


class HttpEmbedding:
    """A learned embedding model served over an OpenAI-compatible ``/v1/embeddings`` endpoint.

    ``url`` is the endpoint (e.g. ``http://localhost:8080/v1/embeddings``), ``model`` the model id
    it serves, and ``dim`` the width that model returns — the catalog's vector column is created at
    that width, so a mismatch between the served model and ``dim`` is an error, not a silent
    truncation. Vectors are L2-normalized here, so cosine similarity is a dot product for every
    provider alike.
    """

    def __init__(
        self,
        url: str,
        *,
        model: str,
        dim: int,
        api_key: str | None = None,
        timeout: float = _HTTP_TIMEOUT_S,
    ) -> None:
        self.url = url
        self.model = model
        self._dim = dim
        self._api_key = api_key
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"http:{self.model}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            return tuple([0.0] * self._dim)
        body = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload: dict[str, Any] = json.loads(response.read())
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            raise EmbeddingError(f"embedding request to {self.url} failed: {exc}") from exc

        try:
            vector = [float(value) for value in payload["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"malformed embedding response from {self.url}") from exc
        if len(vector) != self._dim:
            raise EmbeddingError(
                f"{self.model} returned a {len(vector)}-d vector; provider is configured for "
                f"{self._dim}-d (the catalog's vector column width)"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(vector) if norm == 0.0 else tuple(value / norm for value in vector)


def default_provider() -> EmbeddingProvider:
    """The configured provider: :class:`HttpEmbedding` if ``HUB_EMBEDDING_URL`` is set, else hash.

    Deployment picks the backend by environment (``HUB_EMBEDDING_URL``, ``HUB_EMBEDDING_MODEL``,
    ``HUB_EMBEDDING_DIM``, ``HUB_EMBEDDING_API_KEY``); with none set, the offline hashing default
    keeps tier-1 working with no service to run (hub.md principle 7, 9).
    """
    url = os.environ.get("HUB_EMBEDDING_URL")
    if not url:
        return HashingEmbedding()
    return HttpEmbedding(
        url,
        model=os.environ.get("HUB_EMBEDDING_MODEL", "text-embedding-3-small"),
        dim=int(os.environ.get("HUB_EMBEDDING_DIM", "1536")),
        api_key=os.environ.get("HUB_EMBEDDING_API_KEY"),
    )
