#!/usr/bin/env python
"""Offline dev signer for a SafetySpec — a thin shim over ``astro-mine guard sign`` (GUARD-05).

The signing logic now lives **in the package** (:mod:`astro_mine.guard.cli`) so it is reachable from
an installed wheel — ``scripts/`` is not packaged. This wrapper is kept only so the old invocation
keeps working; it delegates to the CLI and duplicates nothing. Prefer:

    astro-mine guard sign anchor            # sign the shipped anchor spec's content hash
    astro-mine guard sign my.safety.yaml --verify --pub key.pub.pem

The signature is what a signature-requiring registry / the fail-closed signed-load gate
(:mod:`astro_mine.guard.spec.signed`) then verifies. Untrusted authoring tooling — signing never
happens in, and is never trusted by, the Rust TCB (production trusted-key distribution is decided
with Hub, RFC-0004).
"""

from __future__ import annotations

import sys

from astro_mine.guard.cli import main

if __name__ == "__main__":
    # Preserve the old positional-spec surface: `sign_spec.py <spec> [--key ...] [--verify]`.
    sys.exit(main(["sign", *sys.argv[1:]]))
