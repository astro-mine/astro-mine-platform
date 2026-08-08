"""Leaderboard wire models (RM-P0-BENCH-06; bench.md §9).

The typed request/response surface of the minimal leaderboard: a :class:`SubmissionRequest`
(submit-policy-we-run intake), the content-addressed :class:`Submission` record it produces
(the held-out scorecard + integrity verdict), and the ranked :class:`LeaderboardEntry` rows.
Pydantic v2 models (conventions.md §3); Bench owns them — they are not a Core schema.

Backlog: RM-P0-BENCH-06 — astro-mine-bench#6
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HubSubmissionRequest",
    "Integrity",
    "LeaderboardEntry",
    "MetricScore",
    "Submission",
    "SubmissionRequest",
]

#: A submission's integrity verdict from sampled re-execution (bench.md §9).
Integrity = Literal["verified", "flagged"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SubmissionRequest(BaseModel):
    """A leaderboard submission: which scenario, and the policy to run (submit-policy-we-run).

    ``policy_ref`` is a ``"module:attribute"`` reference the server runs on the scenario's held-out
    seeds. It is **untrusted code**: since bench#30 the server does *not* import it — the reference
    is shape-checked at the edge and resolved (imported) only inside the sandboxed eval worker,
    out-of-process, with no network egress and hard CPU/memory/time caps (bench.md §9). The caller
    must present a valid OIDC bearer token (bench#29).
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    policy_ref: str = Field(min_length=1)
    method: str | None = None
    author: str | None = None


class HubSubmissionRequest(BaseModel):
    """A community submission referenced **only by Hub digest** (RM-P1-BENCH-10; bench.md §6).

    ``hub_ref`` is a Hub reference — a ``name:version`` tag or a ``sha256:`` image-manifest digest —
    that Bench resolves from Hub, **verifies fail-closed** (content address, then cosign signature +
    SLSA provenance + SBOM — bench#29), whose Core plugin manifest is validated against the scenario
    interface, and which is then run under submit-policy-we-run **inside a sandbox** (bench#30). No
    policy bytes are uploaded: the artifact is authenticated by content hash and by signature.

    ``method``/``author`` are display metadata only. There is deliberately **no ``identity``
    field**: the submitter's identity comes from the verified OIDC bearer token, and rate limits,
    quotas, job tickets, and audit records are all keyed on that (bench#29). The pre-bench#29 model
    carried a client-supplied ``identity`` that keyed the rate limiter — so a submitter could reset
    their own quota by editing a JSON field.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    hub_ref: str = Field(min_length=1)
    method: str | None = None
    author: str | None = None


class MetricScore(_Model):
    """One metric's aggregate on the held-out seeds — the transparent, per-metric record."""

    metric: str
    unit: str
    direction: str
    aggregation: str
    value: float | None
    dispersion: float | None
    n: int


class Submission(_Model):
    """A content-addressed leaderboard record: the held-out scorecard + its integrity verdict.

    ``submission_id`` is the ``sha256:`` digest of ``(scenario_id, policy_ref, scorecard_hash)``,
    so re-submitting an identical policy is idempotent and a result is reproducible from its
    provenance (conventions.md §5). ``integrity`` is the sampled-re-execution verdict (bench.md §9).
    """

    submission_id: str
    scenario_id: str
    policy_ref: str
    method: str | None
    author: str | None
    scorecard_hash: str
    #: The identity of the runner that produced this scorecard — ``"fixture/0.1.0"`` for the
    #: dependency-clean reference fixture, or an injected Sim runner's own id (bench.md §10; G1.1).
    #: Already folded into ``scorecard_hash`` (so it does not re-address the submission), but held
    #: as a **first-class field** so a downstream reader — a leaderboard row, a paper — can tell a
    #: fixture score from a simulated one **by provenance, not by value** (G1.8). Additive and, like
    #: ``trace_hash``, deliberately **excluded from** ``submission_id``.
    runner: str
    integrity: Integrity
    scores: tuple[MetricScore, ...]
    #: The submission's provenance handle — a Hub image-manifest digest for a community submission
    #: resolved by digest (RM-P1-BENCH-10), or ``None`` for a local ``policy_ref`` submission.
    source: str | None = None
    #: The ``sha256:`` digest of the stored :class:`ProvenanceBundle` — the full lineage this entry
    #: is byte-for-byte reproducible from (bench.md §5). ``None`` for the P0 in-line path.
    provenance_hash: str | None = None
    #: The ``sha256:`` digest of this entry's stored MCAP episode replay (the object-store key View
    #: fetches to render an episode; RM-P1-BENCH-12). ``None`` until a replay is attached. Additive
    #: and **excluded from** ``submission_id`` (computed in ``build_submission``), so attaching a
    #: replay never changes the content address of an already-catalogued submission.
    trace_hash: str | None = None


class LeaderboardEntry(_Model):
    """One ranked row of a scenario leaderboard, ordered by the scenario's primary metric."""

    rank: int
    submission_id: str
    method: str | None
    author: str | None
    integrity: Integrity
    primary_metric: str
    primary_value: float | None
    primary_unit: str
    #: The submission's Hub digest (community submission) or ``None`` (local ``policy_ref``).
    source: str | None = None
    #: The stored :class:`ProvenanceBundle` digest — the lineage the row is reproducible from.
    provenance_hash: str | None = None
