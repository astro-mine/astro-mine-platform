"""The OPA engine's transport + fail-closed behaviour, offline (RM-P1-HUB-05).

The *policy* conformance — that the Rego bundle and the Python evaluator decide identically — needs
a real OPA and lives in ``tests/integration/test_opa_policy.py``. What is testable offline (and what
must never regress) is the engine's **failure surface**: OPA absent, unreachable, exiting non-zero,
or returning an undefined/malformed decision must all raise, because a policy that cannot be
evaluated **denies** the download. There is no code path here that allows on error (hub.md §9;
``LUNAR-SR-001``).
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from astro_mine.hub.index import CatalogEntry
from astro_mine.hub.policy import (
    DownloadRequest,
    OpaPolicyEngine,
    OpaUnavailable,
    opa_engine_from_env,
    policy_data,
)
from astro_mine.hub.policy._opa import OPA_DECISION_PATH

from .conftest import DIGEST as _DIGEST
from .conftest import POLICY_CASES, make_manifest

ALLOW = {"allow": True, "code": "allowed", "reason": "allowed", "version": "1.0.0"}
DENY = {
    "allow": False,
    "code": "license_denied",
    "reason": "license GPL-3.0-only is not permitted",
    "version": "1.0.0",
}


class _FakeOpa:
    """An OPA sidecar (``POST /v1/data/<path>``) — the transport contract, served over real HTTP."""

    def __init__(self, decision: dict[str, Any] | None, *, undefined: bool = False) -> None:
        self.decision = decision
        self.undefined = undefined
        self.inputs: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                assert self.path == f"/v1/data/{OPA_DECISION_PATH}"
                length = int(self.headers.get("Content-Length") or 0)
                server.inputs.append(json.loads(self.rfile.read(length))["input"])
                payload = {} if server.undefined else {"result": server.decision}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _FakeOpa:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"


def _entry(license: str | None = "Apache-2.0") -> CatalogEntry:
    return CatalogEntry(
        manifest=make_manifest(license=license), digest=_DIGEST, publisher="p", namespace="open"
    )


def test_sidecar_decision_round_trip() -> None:
    with _FakeOpa(ALLOW) as opa:
        engine = OpaPolicyEngine(url=opa.url)
        assert engine.name == "opa"
        assert engine.version == policy_data().version

        decision = engine.evaluate(_entry(), DownloadRequest())
        assert decision.allowed is True
        assert decision.code == "allowed"
        assert decision.engine == "opa"
        assert decision.policy_version == "1.0.0"
        assert decision.reference == "pol:1.0.0"

        # the input document OPA saw is the shared conformance contract, not an OPA-specific shape
        assert opa.inputs[0]["license"] == "Apache-2.0"
        assert opa.inputs[0]["require_verified"] is False


def test_sidecar_denial_is_a_denial_not_an_error() -> None:
    with _FakeOpa(DENY) as opa:
        decision = OpaPolicyEngine(url=opa.url).evaluate(_entry("GPL-3.0-only"), DownloadRequest())
        assert decision.allowed is False and decision.code == "license_denied"


def test_undefined_result_fails_closed() -> None:
    """No rule produced a decision — deny. (An "undefined" OPA result must never read as allow.)"""
    with (
        _FakeOpa(None, undefined=True) as opa,
        pytest.raises(OpaUnavailable, match="no decision"),
    ):
        OpaPolicyEngine(url=opa.url).evaluate(_entry(), DownloadRequest())


def test_malformed_decision_fails_closed() -> None:
    with (  # a decision with no code/reason/version
        _FakeOpa({"allow": True}) as opa,
        pytest.raises(OpaUnavailable, match="missing"),
    ):
        OpaPolicyEngine(url=opa.url).evaluate(_entry(), DownloadRequest())


def test_unreachable_sidecar_fails_closed() -> None:
    engine = OpaPolicyEngine(url="http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(OpaUnavailable, match="failed"):
        engine.evaluate(_entry(), DownloadRequest())


def test_no_opa_at_all_refuses_to_construct(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub configured for OPA fails loudly rather than silently using another policy engine."""
    monkeypatch.setattr("astro_mine.hub.policy._opa.shutil.which", lambda _: None)
    with pytest.raises(OpaUnavailable, match="no OPA available"):
        OpaPolicyEngine()


def test_binary_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """`opa eval --bundle` — parsed from the real CLI's JSON shape (the binary runs in CI)."""
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["input"] = json.loads(kwargs["input"])
        payload = {"result": [{"expressions": [{"value": ALLOW}]}]}
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr("astro_mine.hub.policy._opa.shutil.which", lambda _: "/usr/bin/opa")
    monkeypatch.setattr(subprocess, "run", fake_run)

    decision = OpaPolicyEngine().evaluate(_entry(), DownloadRequest())
    assert decision.allowed is True and decision.engine == "opa"
    assert captured["argv"][:3] == ["/usr/bin/opa", "eval", "--format=json"]
    assert captured["argv"][-1] == "data.astro_mine.hub.download.decision"
    assert captured["input"]["reference"] == "pol:1.0.0"


@pytest.mark.parametrize(
    ("stdout", "returncode", "match"),
    [
        ("", 1, "exited 1"),
        ("not json", 0, "no decision"),
        ('{"result": []}', 0, "no decision"),  # OPA's shape for an undefined document
        ('{"result": [{"expressions": [{"value": "nope"}]}]}', 0, "non-object"),
    ],
)
def test_binary_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int, match: str
) -> None:
    monkeypatch.setattr("astro_mine.hub.policy._opa.shutil.which", lambda _: "/usr/bin/opa")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, returncode, stdout, "boom"),
    )
    with pytest.raises(OpaUnavailable, match=match):
        OpaPolicyEngine().evaluate(_entry(), DownloadRequest())


def test_binary_not_executable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("astro_mine.hub.policy._opa.shutil.which", lambda _: "/usr/bin/opa")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("exec format error")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(OpaUnavailable, match="opa eval failed"):
        OpaPolicyEngine().evaluate(_entry(), DownloadRequest())


def test_engine_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither env var set, Hub stays on the Python evaluator — the offline default."""
    monkeypatch.delenv("HUB_OPA_URL", raising=False)
    monkeypatch.delenv("HUB_POLICY_ENGINE", raising=False)
    assert opa_engine_from_env() is None

    monkeypatch.setenv("HUB_OPA_URL", "http://opa.internal:8181")
    engine = opa_engine_from_env()
    assert engine is not None and engine.url == "http://opa.internal:8181"

    monkeypatch.delenv("HUB_OPA_URL")
    monkeypatch.setenv("HUB_POLICY_ENGINE", "opa")
    monkeypatch.setattr("astro_mine.hub.policy._opa.shutil.which", lambda _: "/usr/bin/opa")
    assert opa_engine_from_env() is not None


def test_every_conformance_case_produces_an_input_document() -> None:
    """Whatever the case, the engine hands OPA a well-formed input (nothing unrepresentable)."""
    with _FakeOpa(ALLOW) as opa:
        engine = OpaPolicyEngine(url=opa.url)
        for case in POLICY_CASES:
            engine.evaluate(case.entry, case.request)
    assert len(opa.inputs) == len(POLICY_CASES)
    assert all("allowed_licenses" in document for document in opa.inputs)
