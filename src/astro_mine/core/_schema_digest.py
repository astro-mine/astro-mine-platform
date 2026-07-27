"""The Core schema digest — GENERATED, do not edit by hand.

Regenerate with::

    uv run python scripts/build_schema_bundle.py --update-digest

The digest is computed over the *full* schema source set — including the ``.proto`` sources
under ``schemas/proto/``, which live at the repo root and are **not** packaged in the wheel.
That is why this is a committed constant rather than something recomputed at import time: an
installed Core cannot see those files, so it could not reproduce the digest if it tried, and
a filesystem walk relative to ``__file__`` would yield a plausible-but-wrong value in a wheel
while looking perfectly correct in this repo (#55).
"""

SCHEMA_DIGEST = "sha256:2ebc6353bda4ecd0ed14b39ef04747b84a8fa79f8a094146f74ee027cbf07980"
