"""OPA-style policy evaluation: quotas, embargo, and authoring rights (bench#29; bench.md §9).

bench.md §9: *"RBAC via **OPA** for submission quotas, embargo control, and metric/scenario
authoring rights"*. This module is that authorization layer, shaped exactly like an OPA decision so
the two implementations are interchangeable:

- an :class:`AuthorizationRequest` renders to the OPA **input document** (:meth:`~
  AuthorizationRequest.to_input`) — ``{"principal": …, "action": …, "resource": …, "context": …}``;
- a :class:`Decision` is the OPA **result document** — ``{"allow": bool, "reason": str}``;
- :class:`RbacPolicyEngine` evaluates that document in-process (the default, and what the local and
  single-node deployments run), while :class:`OpaPolicyEngine` POSTs the *same* document to a real
  OPA sidecar and reads the *same* result back. ``policy/bench.rego`` is the Rego source of the
  built-in rules, so the sidecar and the in-process engine decide alike.

Three rule families, per bench.md §9:

1. **RBAC** — a role grants a set of actions; an action no role of the principal grants is denied.
2. **Submission quotas** — each role carries a cap on submissions per rate-limit window; the
   ``submissions_in_window`` count in the context (keyed on the *authenticated* subject, never a
   client-supplied identity) is checked against it. This is "compute-for-score" back-pressure —
   bench.md §8/§9.
3. **Embargo control** — an embargoed scenario is not open season: submitting to one, or reading its
   held-out material, requires the ``embargo:read`` right, which only maintainers and admins hold.

Everything **defaults to deny**: an unknown action, an unknown role, an engine error, an unreachable
sidecar, or an OPA response that does not explicitly say ``allow: true`` all produce a denial
(conventions.md §9 — auditable policy, fail-closed).

Backlog: bench#29 — https://github.com/astro-mine/astro-mine-bench/issues/29
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from astro_mine.bench.leaderboard._auth import Principal

__all__ = [
    "DEFAULT_QUOTAS",
    "DEFAULT_ROLE_ACTIONS",
    "EMBARGOED_SCENARIOS_ENV",
    "OPA_URL_ENV",
    "Action",
    "AuthorizationError",
    "AuthorizationRequest",
    "Decision",
    "OpaPolicyEngine",
    "PolicyEngine",
    "RbacPolicyEngine",
    "Role",
    "policy_engine_from_env",
]

#: Env vars: point at an OPA sidecar, and declare which scenarios are under embargo.
OPA_URL_ENV = "ASTRO_MINE_BENCH_OPA_URL"
EMBARGOED_SCENARIOS_ENV = "ASTRO_MINE_BENCH_EMBARGOED_SCENARIOS"


class Action(StrEnum):
    """Every authorization decision the hosted leaderboard makes (bench.md §9)."""

    #: Submit a local ``policy_ref`` to be run on the held-out seeds (``POST /submissions``).
    SUBMISSION_CREATE = "submission:create"
    #: Submit a community artifact by Hub digest (``POST /submissions/hub``).
    SUBMISSION_CREATE_HUB = "submission:create_hub"
    #: Mutate the board: retract or re-rank an entry (``DELETE /submissions/{id}``).
    RANKING_MUTATE = "ranking:mutate"
    #: Publish/curate a ScenarioSpec in the hosted zoo catalog (``POST /scenarios``).
    SCENARIO_AUTHOR = "scenario:author"
    #: Publish a community metric plugin against a scenario's metric set.
    METRIC_AUTHOR = "metric:author"
    #: Reach embargoed material — a held-out seed set, or an embargoed scenario.
    EMBARGO_READ = "embargo:read"
    #: Read the authN/authZ + verification audit trail (``GET /audit``).
    AUDIT_READ = "audit:read"


class Role(StrEnum):
    """The roles the IdP asserts in the token's roles claim."""

    SUBMITTER = "submitter"
    MAINTAINER = "maintainer"
    ADMIN = "admin"


#: Which actions each role grants. A principal with no known role grants nothing (default-deny).
DEFAULT_ROLE_ACTIONS: dict[Role, frozenset[Action]] = {
    Role.SUBMITTER: frozenset({Action.SUBMISSION_CREATE, Action.SUBMISSION_CREATE_HUB}),
    Role.MAINTAINER: frozenset(
        {
            Action.SUBMISSION_CREATE,
            Action.SUBMISSION_CREATE_HUB,
            Action.SCENARIO_AUTHOR,
            Action.METRIC_AUTHOR,
            Action.EMBARGO_READ,
        }
    ),
    Role.ADMIN: frozenset(Action),
}

#: Submissions allowed per rate-limit window, per role — the per-user quota bench.md §9 requires.
#: A maintainer runs regression sweeps; an admin is uncapped (``None``).
DEFAULT_QUOTAS: dict[Role, int | None] = {
    Role.SUBMITTER: 20,
    Role.MAINTAINER: 200,
    Role.ADMIN: None,
}

#: The actions a submission quota applies to.
_QUOTA_ACTIONS = frozenset({Action.SUBMISSION_CREATE, Action.SUBMISSION_CREATE_HUB})


class Decision(BaseModel):
    """An authorization decision — the OPA result document (``{"allow": …, "reason": …}``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow: bool
    reason: str = ""


class AuthorizationError(Exception):
    """The principal may not perform the action — rejected with 403, and always audit-logged."""

    def __init__(self, decision: Decision, *, action: Action, resource: str) -> None:
        super().__init__(decision.reason or f"{action} denied on {resource!r}")
        self.decision = decision
        self.action = action
        self.resource = resource


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    """The OPA **input document**: who wants to do what, to what, in what circumstances."""

    principal: Principal
    action: Action
    resource: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_input(self) -> dict[str, Any]:
        """Render the OPA ``input`` document — the same shape both engines evaluate."""
        return {
            "principal": {
                "subject": self.principal.subject,
                "issuer": self.principal.issuer,
                "roles": list(self.principal.roles),
                "scopes": list(self.principal.scopes),
            },
            "action": str(self.action),
            "resource": self.resource,
            "context": dict(self.context),
        }


class PolicyEngine(Protocol):
    """Evaluates an :class:`AuthorizationRequest` to a :class:`Decision` (default-deny)."""

    def evaluate(self, request: AuthorizationRequest) -> Decision:
        """Decide ``request``. MUST return ``allow=False`` on any error — never raise past."""
        ...


class RbacPolicyEngine:
    """The built-in policy engine — RBAC + quotas + embargo (see ``policy/bench.rego``).

    Evaluates the same input document an OPA sidecar would, with the same rules, so a deployment can
    move to :class:`OpaPolicyEngine` without a behavioural change. ``embargoed_scenarios`` are the
    scenarios whose held-out material is still under embargo — submitting to one needs the
    ``embargo:read`` right.
    """

    def __init__(
        self,
        *,
        role_actions: Mapping[Role, frozenset[Action]] | None = None,
        quotas: Mapping[Role, int | None] | None = None,
        embargoed_scenarios: frozenset[str] = frozenset(),
    ) -> None:
        self._role_actions = dict(role_actions or DEFAULT_ROLE_ACTIONS)
        self._quotas = dict(quotas or DEFAULT_QUOTAS)
        self._embargoed = frozenset(embargoed_scenarios)

    def _roles(self, principal: Principal) -> tuple[Role, ...]:
        """The principal's *known* roles; an unrecognized role name grants nothing."""
        known: list[Role] = []
        for name in principal.roles:
            try:
                known.append(Role(name))
            except ValueError:
                continue  # an unknown role is not an error, it is simply no grant (default-deny)
        return tuple(known)

    def _quota(self, roles: tuple[Role, ...]) -> int | None:
        """The most generous quota the principal's roles carry (``None`` = uncapped)."""
        caps = [self._quotas.get(role, 0) for role in roles]
        if any(cap is None for cap in caps):
            return None
        return max((cap for cap in caps if cap is not None), default=0)

    def evaluate(self, request: AuthorizationRequest) -> Decision:
        """RBAC, then the submission quota, then embargo — the first failure denies."""
        roles = self._roles(request.principal)
        if not roles:
            return Decision(
                allow=False,
                reason=f"principal {request.principal.subject!r} carries no known role",
            )

        granted = frozenset[Action]().union(
            *(self._role_actions.get(role, frozenset()) for role in roles)
        )
        if request.action not in granted:
            return Decision(
                allow=False,
                reason=(
                    f"role(s) {[str(r) for r in roles]} do not grant {request.action} "
                    f"on {request.resource!r}"
                ),
            )

        if request.action in _QUOTA_ACTIONS:
            quota = self._quota(roles)
            used = int(request.context.get("submissions_in_window", 0))
            if quota is not None and used > quota:
                return Decision(
                    allow=False,
                    reason=(
                        f"submission quota exhausted: {used} > {quota} in the current window for "
                        f"role(s) {[str(r) for r in roles]}"
                    ),
                )

        # Embargo control: an embargoed scenario is reachable only with the embargo:read right.
        scenario = str(request.context.get("scenario_id") or request.resource)
        if scenario in self._embargoed and Action.EMBARGO_READ not in granted:
            return Decision(
                allow=False,
                reason=(
                    f"scenario {scenario!r} is under embargo; {Action.EMBARGO_READ} is required"
                ),
            )

        return Decision(allow=True, reason=f"{request.action} granted by {[str(r) for r in roles]}")


class OpaPolicyEngine:
    """Delegate the decision to an external **Open Policy Agent** sidecar (bench.md §9).

    POSTs the :meth:`AuthorizationRequest.to_input` document to OPA's data API and reads back
    ``{"result": {"allow": …, "reason": …}}``. **Fail-closed on everything**: a network error, a
    timeout, a non-200, a malformed body, or an undefined rule (OPA returns ``{}`` when no rule
    matched) all deny — an authorization service that cannot answer must never mean "yes".
    """

    def __init__(self, url: str, *, http: Any = None, timeout: float = 2.0) -> None:
        if not url:
            raise ValueError("OpaPolicyEngine needs the URL of an OPA decision endpoint")
        self._url = url
        self._http = http
        self._timeout = timeout

    def evaluate(self, request: AuthorizationRequest) -> Decision:
        import httpx

        client = self._http if self._http is not None else httpx.Client(timeout=self._timeout)
        try:
            response = client.post(self._url, json={"input": request.to_input()})
            response.raise_for_status()
            body: Any = response.json()
        except Exception as exc:
            return Decision(allow=False, reason=f"OPA policy evaluation failed: {exc}")
        finally:
            if self._http is None:
                client.close()

        result = body.get("result") if isinstance(body, Mapping) else None
        if not isinstance(result, Mapping):
            # OPA returns an empty document for an undefined rule — that is a denial, not a pass.
            return Decision(allow=False, reason="OPA returned no decision (undefined policy rule)")
        allow = result.get("allow")
        if allow is not True:
            return Decision(
                allow=False, reason=str(result.get("reason") or f"{request.action} denied by OPA")
            )
        return Decision(allow=True, reason=str(result.get("reason") or "allowed by OPA"))


def policy_engine_from_env(env: Mapping[str, str] | None = None) -> PolicyEngine:
    """Select the policy engine from the environment: an OPA sidecar if configured, else RBAC.

    Both enforce the same rules; the sidecar exists so an operator can evolve policy in Rego without
    redeploying Bench.
    """
    environment = os.environ if env is None else env
    embargoed = frozenset(
        part.strip()
        for part in environment.get(EMBARGOED_SCENARIOS_ENV, "").split(",")
        if part.strip()
    )
    url = environment.get(OPA_URL_ENV)
    if url:
        return OpaPolicyEngine(url)
    return RbacPolicyEngine(embargoed_scenarios=embargoed)
