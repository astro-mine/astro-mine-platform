# SPDX-License-Identifier: Apache-2.0
"""Leaderboard submission — the CLI's write path (G2.14; RM-P1-BENCH-10; bench.md §6).

The public surface of :mod:`astro_mine.bench.submit._client`. Needs the optional ``[submit]``
extra (``httpx``); the base package stays install-light so ``score`` and ``list`` keep working
offline with no account (CX-LOCAL).
"""

from __future__ import annotations

from astro_mine.bench.submit._client import (
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_WAIT_TIMEOUT_S,
    TOKEN_ENV,
    SubmitError,
    await_job,
    get_job,
    get_submission,
    is_rejected,
    poll_job,
    rank_of,
    read_token,
    submit_hub,
    submit_policy,
)

__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_WAIT_TIMEOUT_S",
    "TOKEN_ENV",
    "SubmitError",
    "await_job",
    "get_job",
    "get_submission",
    "is_rejected",
    "poll_job",
    "rank_of",
    "read_token",
    "submit_hub",
    "submit_policy",
]
