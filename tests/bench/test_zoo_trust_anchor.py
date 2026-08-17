# SPDX-License-Identifier: Apache-2.0
"""Every zoo-pinned artifact verifies against the **published** trust anchor.

This is the check whose absence let a real defect through. The §13 artifact-name migration
(astro-mine-platform#34) re-published all ten anchor artifacts, and the sweep signed them with
`files/hub-registry/keys/anchor-dev.key.pem` — the development key that seeds the workspace store —
because that is the key the local store uses and nothing asked the question. The migration was
verified for names, digests, reproducibility and the full test suite, and every one of those passed.
None of them looks at *which key signed the bytes*.

The consequence is specific and severe: `anchor-signing.pub` is the committed, published trust
anchor, and Hub's verification is **fail-closed** by design (`hub.md` §2.3, `LUNAR-SR-002`). An
artifact signed with any other key does not merely carry a weaker guarantee — it does not resolve at
all for a consumer who pins the anchor. Publishing such an artifact to the org registry would be
worse than publishing nothing, because the zoo pins it by digest and the failure surfaces at pull.

So: a digest the zoo pins must verify against the key the project publishes. Not against *a* key,
and not merely "carries a signature" — against **that** key.

**Why this needs the store, and what it does when it cannot reach one.** Verification needs the
artifact bytes, and those live in an OCI registry rather than in this repository — the workspace
store, or `ghcr.io/astro-mine`. A clean clone has neither. Rather than skip in that case, the check
is pointed at whatever `$ASTRO_MINE_HUB_REGISTRY` names — the same variable
`astro_mine.bench.content`, the eval worker and `sim.__main__` already resolve their store through —
and says loudly when it has verified nothing. A signing check that quietly passes on a machine with
no artifacts is precisely the shape of the hole it was written to close.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ZOO = REPO / "src" / "astro_mine" / "bench" / "zoo"
TRUST_ANCHOR = REPO / "anchor-signing.pub"
STORE_ENV = "ASTRO_MINE_HUB_REGISTRY"


def _pinned_digests() -> dict[str, str]:
    """``content id -> content_hash`` across every zoo scenario, deduplicated."""
    pinned: dict[str, str] = {}
    for path in sorted(ZOO.rglob("scenario.json")):
        content = json.loads(path.read_text(encoding="utf-8")).get("content") or {}
        refs = []
        if isinstance(content.get("world"), dict):
            refs.append(content["world"])
        for key in ("fleet", "prospect"):
            refs.extend(content.get(key) or [])
        if isinstance(content.get("link"), dict):
            refs.append(content["link"])
        for ref in refs:
            pinned[str(ref["id"])] = str(ref["content_hash"])
    return pinned


def _store() -> pathlib.Path | None:
    configured = os.environ.get(STORE_ENV)
    if not configured:
        return None
    path = pathlib.Path(configured).expanduser()
    return path if (path / "oci-layout").is_file() else None


STORE = _store()

pytestmark = pytest.mark.skipif(
    STORE is None,
    reason=(
        f"no OCI-layout store at ${STORE_ENV}. THIS IS NOT A PASS — no zoo-pinned artifact had its "
        f"signature checked against the published trust anchor on this run. Point {STORE_ENV} at a "
        f"store holding the anchor content (astro-mine-platform#41)."
    ),
)


def test_the_trust_anchor_is_committed() -> None:
    """The anchor is in the repository, so 'which key' is answerable without a secret."""
    assert TRUST_ANCHOR.is_file(), f"{TRUST_ANCHOR.name} is missing — nothing to verify against"
    assert b"PUBLIC KEY" in TRUST_ANCHOR.read_bytes()


def test_every_zoo_pinned_artifact_verifies_against_the_published_anchor() -> None:
    """**The gate.** A dev-key-signed artifact in the zoo fails here, naming it.

    Failure modes this separates, because they need different fixes: an artifact the store does not
    hold at all (mirror it, or point at the right store) versus one it holds but which the anchor
    does not vouch for (re-sign it — signatures are OCI *referrers*, so re-signing does not move the
    digest and the zoo's pins stay valid).
    """
    from astro_mine.hub.registry import Registry
    from astro_mine.hub.supply_chain import verify

    assert STORE is not None  # narrowed by pytestmark
    registry = Registry(STORE)
    anchor = TRUST_ANCHOR.read_bytes()

    pinned = _pinned_digests()
    assert pinned, "found no pinned content — the discovery above has broken, not the zoo"

    absent: list[str] = []
    unvouched: list[str] = []
    for name, digest in sorted(pinned.items()):
        try:
            registry.verify(digest)
        except Exception:
            absent.append(f"{name}@{digest[:19]}")
            continue
        try:
            verify(registry, digest, trusted_public_key_pem=anchor)
        except Exception:
            unvouched.append(f"{name}@{digest[:19]}")

    assert not unvouched, (
        f"{len(unvouched)} zoo-pinned artifact(s) do not verify against the published trust "
        f"anchor: "
        f"{unvouched}. Hub verifies fail-closed, so these do not resolve for any consumer pinning "
        f"anchor-signing.pub. Re-sign them with the org key — signatures ride as referrers, so the "
        f"digest and the zoo's pins are unaffected (astro-mine-platform#41)."
    )
    assert not absent, (
        f"{len(absent)} zoo-pinned artifact(s) are not in this store: {absent}. Either the store "
        f"is "
        f"the wrong one, or the content was never mirrored to it."
    )
