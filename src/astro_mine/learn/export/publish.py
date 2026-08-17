# SPDX-License-Identifier: Apache-2.0
"""Publish an exported PolicyPackage to a content-addressed store (RM-P1-LEARN-05; §5).

The ONNX graph + typed sidecar are written to a **content-addressed** store keyed by the
graph digest, so Bench resolves the artifact by content hash and Mind/Guard load it via
:class:`~astro_mine.core.policy.OnnxPolicy` — the M1.2 flywheel — **without Learn depending on
Hub**. The hosted Hub push (cosign OCI; hub.md §9) is a thin *optional* handoff: the caller
passes a ``publisher`` callable (the astro-mine-hub client adapts to it), so Hub internals stay
out of Learn's dependency tree (RM-P1-HUB-* out of scope here).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from astro_mine.core.policy.model import PolicyPackageDocument
from astro_mine.learn.export.package import ExportedPolicy

__all__ = ["PublishedPolicy", "publish"]

_ONNX_FILENAME = "model.onnx"
_SIDECAR_FILENAME = "policy_package.json"


@dataclass(frozen=True)
class PublishedPolicy:
    """A published artifact: its graph digest and the on-disk ONNX + sidecar paths."""

    digest: str
    onnx_path: Path
    sidecar_path: Path
    document: PolicyPackageDocument


def publish(
    exported: ExportedPolicy,
    store_dir: str | Path,
    *,
    publisher: Callable[[PublishedPolicy], None] | None = None,
) -> PublishedPolicy:
    """Write ``exported`` to a content-addressed store under ``store_dir`` and return its handle.

    The graph digest (``sha256:<hex>``) keys the layout ``<store_dir>/<hex>/{model.onnx,
    policy_package.json}``; the sidecar records the graph's ``file://`` URI so a consumer can
    fetch the bytes. ``publisher``, if given, is invoked with the written artifact — the
    optional Hub handoff (the astro-mine-hub client) — after the local write succeeds.

    ``store_dir`` may be relative; it is resolved to an absolute path first. The sidecar has to
    record a ``file://`` URI, and :meth:`Path.as_uri` refuses a relative path — so a relative
    store used to fail *between* the two writes, leaving a digest directory holding the graph
    with no sidecar beside it (#33). A half-entry is worse than no entry: it still resolves by
    digest, and what a consumer gets is a model with no IO signature, no assumptions and no
    provenance.

    The entry is materialized **atomically**. Everything that can fail — the URI, the located
    document, the serialized sidecar — is computed before a single byte is written, and the two
    files are then built in a temporary directory beside the destination and renamed into place.
    So an interrupted publish leaves nothing rather than half of a ``PolicyPackage``."""
    digest = exported.digest
    key = digest.split(":", 1)[-1]  # filesystem-safe: drop the "sha256:" algorithm prefix
    # Resolve before anything else: `as_uri()` below needs an absolute path, and resolving once
    # here means every path this function records or returns is unambiguous.
    store_root = Path(store_dir).resolve()
    dest = store_root / key
    onnx_path = dest / _ONNX_FILENAME
    sidecar_path = dest / _SIDECAR_FILENAME

    # --- everything fallible, before the first write --------------------------------------------
    #
    # Stamp the local URI onto a copy of the document so the stored sidecar is self-locating;
    # the digest (content identity) is unchanged — it hashes the graph bytes, not the sidecar.
    # The URI names the *final* location, not the temporary one the bytes are staged through.
    package = exported.document.policy_package
    located = package.model_copy(
        update={"onnx_model": package.onnx_model.model_copy(update={"uri": onnx_path.as_uri()})}
    )
    document = exported.document.model_copy(update={"policy_package": located})
    sidecar_bytes = json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True).encode(
        "utf-8"
    )

    # --- materialize ----------------------------------------------------------------------------
    store_root.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Already published. The store is content-addressed, so an existing entry under this key
        # holds these exact graph bytes; rewriting in place is idempotent and keeps a re-publish
        # cheap. (Only the sidecar's `uri` could differ, if the store moved — refreshing it is the
        # right answer, since the old URI no longer locates anything.)
        onnx_path.write_bytes(exported.onnx_bytes)
        sidecar_path.write_bytes(sidecar_bytes)
    else:
        staging = Path(tempfile.mkdtemp(dir=store_root, prefix=f".{key}."))
        try:
            (staging / _ONNX_FILENAME).write_bytes(exported.onnx_bytes)
            (staging / _SIDECAR_FILENAME).write_bytes(sidecar_bytes)
            # Same filesystem by construction (staging is inside store_root), so this is atomic.
            os.replace(staging, dest)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            # A concurrent publisher of the same digest wins the rename; its bytes are ours, so
            # the entry is correct either way. Anything else is a real failure.
            if not (onnx_path.exists() and sidecar_path.exists()):
                raise

    published = PublishedPolicy(
        digest=digest, onnx_path=onnx_path, sidecar_path=sidecar_path, document=document
    )
    if publisher is not None:
        publisher(published)
    return published
