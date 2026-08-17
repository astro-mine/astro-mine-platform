# SPDX-License-Identifier: Apache-2.0
"""The OPA/Rego policy engine — governance rules that evolve without a code change (RM-P1-HUB-05).

hub.md §3 makes policy bundles an **extension point** ("admission and download policies are
data-driven Rego bundles, so governance/export-control rules evolve without code changes") and §11
recommends **OPA at the download boundary** against Core capability tags (conventions.md §12).
:class:`OpaPolicyEngine` is that engine, behind the *same*
:class:`~astro_mine.hub.policy.PolicyEngine` call the pure-Python evaluator satisfies — so an
existing caller of :func:`~astro_mine.hub.policy.evaluate` needs no change to be governed by Rego.

Two deployments, one policy (the bundle in ``policy/rego/``):

- **Embedded/CLI** — the ``opa`` binary evaluates the bundle in-process per decision
  (``opa eval --bundle …``). Nothing to run; good for a CI check or a single-node deployment.
- **Sidecar** — an OPA server (``HUB_OPA_URL``) evaluates the bundle it has loaded and hot-reloads
  on a new bundle revision, which is the point: a license or export-control change is a **bundle
  release**, not a redeploy.

**Fail closed.** Every failure mode — OPA absent, unreachable, non-zero exit, malformed or
*undefined* result — raises :class:`OpaUnavailable`. There is no "allow because the engine did not
answer" path: policy that cannot be evaluated denies the download (hub.md §9; ``LUNAR-SR-001``).
The offline tier-1 path never touches this module: it keeps the Python evaluator (principle 7).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from astro_mine.hub.index import CatalogEntry
from astro_mine.hub.policy._policy import (
    BUNDLE_DIR,
    Decision,
    DownloadRequest,
    PolicyData,
    policy_data,
    policy_input,
)

__all__ = ["OpaPolicyEngine", "OpaUnavailable", "opa_engine_from_env"]

#: The bundle's decision document (``package astro_mine.hub.download``).
OPA_DECISION_PATH = "astro_mine/hub/download/decision"

_TIMEOUT_S = 15.0


class OpaUnavailable(Exception):
    """OPA could not evaluate the policy — the gate fails closed rather than defaulting to allow."""


class OpaPolicyEngine:
    """Evaluate the versioned Rego bundle with OPA — binary (``opa eval``) or sidecar (HTTP)."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        url: str | None = None,
        bundle_dir: Path = BUNDLE_DIR,
        timeout: float = _TIMEOUT_S,
    ) -> None:
        """Bind to an OPA sidecar (``url``) or the ``opa`` binary; ``binary`` defaults to ``PATH``.

        Raises :class:`OpaUnavailable` when neither is reachable — a Hub configured for OPA must
        fail loudly at construction rather than silently fall back to a different policy engine.
        """
        self.url = url.rstrip("/") if url else None
        self.bundle_dir = bundle_dir
        self.timeout = timeout
        self._binary = None if self.url else (binary or shutil.which("opa"))
        if self.url is None and self._binary is None:
            raise OpaUnavailable(
                "no OPA available: install the `opa` binary or set HUB_OPA_URL to a sidecar"
            )
        self._data: PolicyData = policy_data(bundle_dir)

    @property
    def name(self) -> str:
        return "opa"

    @property
    def version(self) -> str:
        """The Rego bundle revision — stamped onto every decision for the audit log."""
        return self._data.version

    # -- transports ------------------------------------------------------------------------------

    def _eval_binary(self, document: dict[str, object]) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [
                    str(self._binary),
                    "eval",
                    "--format=json",
                    "--bundle",
                    str(self.bundle_dir),
                    "--stdin-input",
                    f"data.{OPA_DECISION_PATH.replace('/', '.')}",
                ],
                input=json.dumps(document),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OpaUnavailable(f"opa eval failed: {exc}") from exc
        if completed.returncode != 0:
            raise OpaUnavailable(f"opa eval exited {completed.returncode}: {completed.stderr}")
        try:
            payload = json.loads(completed.stdout)
            value = payload["result"][0]["expressions"][0]["value"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise OpaUnavailable(f"opa returned no decision: {completed.stdout!r}") from exc
        if not isinstance(value, dict):
            raise OpaUnavailable(f"opa returned a non-object decision: {value!r}")
        return value

    def _eval_sidecar(self, document: dict[str, object]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}/v1/data/{OPA_DECISION_PATH}",
            data=json.dumps({"input": document}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            raise OpaUnavailable(f"OPA sidecar at {self.url} failed: {exc}") from exc
        value = payload.get("result")
        if not isinstance(value, dict):
            # An *undefined* result means no rule produced a decision — deny, never allow.
            raise OpaUnavailable(f"OPA sidecar returned no decision: {payload!r}")
        return value

    # -- PolicyEngine ----------------------------------------------------------------------------

    def evaluate(self, entry: CatalogEntry, request: DownloadRequest) -> Decision:
        """The Rego bundle's verdict for ``entry`` under ``request`` (identical to the Python one).

        The input document is :func:`~astro_mine.hub.policy._policy.policy_input` — byte-for-byte
        what the Python evaluator sees, which is what makes the two engines conformance-testable.
        """
        document = policy_input(entry, request)
        decision = self._eval_sidecar(document) if self.url else self._eval_binary(document)
        try:
            allowed = bool(decision["allow"])
            code = str(decision["code"])
            reason = str(decision["reason"])
            version = str(decision["version"])
        except KeyError as exc:
            raise OpaUnavailable(f"OPA decision is missing {exc}: {decision!r}") from exc
        return Decision(
            allowed=allowed,
            reference=entry.reference,
            reason=reason,
            code=code,
            policy_version=version,
            engine=self.name,
        )


def opa_engine_from_env() -> OpaPolicyEngine | None:
    """An :class:`OpaPolicyEngine` if the deployment asks for one, else ``None`` (stay on Python).

    Set ``HUB_OPA_URL`` (a sidecar) or ``HUB_POLICY_ENGINE=opa`` (the ``opa`` binary). With neither,
    Hub keeps the pure-Python evaluator — the offline default (hub.md principle 7).
    """
    url = os.environ.get("HUB_OPA_URL")
    if url:
        return OpaPolicyEngine(url=url)
    if os.environ.get("HUB_POLICY_ENGINE", "").lower() == "opa":
        return OpaPolicyEngine()
    return None
