"""Registry credentials (RM-P1-HUB-06): the standard Docker/OCI sources, never a bespoke scheme.

`docker login`, a cloud credential helper, or a CI token env var must all just work — and a missing
or malformed source must degrade to an **anonymous** pull (public artifacts stay frictionless,
hub.md §2 principle 5) rather than failing.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from astro_mine.hub.registry import Credentials, credentials_for
from astro_mine.hub.registry._auth import _from_helper


def _write_config(
    tmp_path: Path, config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_bytes(json.dumps(config).encode())
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient credentials: neither a developer's ~/.docker nor a CI token leaks into a test."""
    for name in (
        "HUB_REGISTRY_TOKEN",
        "HUB_REGISTRY_USERNAME",
        "GITHUB_TOKEN",
        "GITHUB_ACTOR",
        "REGISTRY_AUTH_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "empty"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty"))


def test_no_credentials_is_anonymous() -> None:
    assert credentials_for("ghcr.io") is None


def test_token_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUB_REGISTRY_TOKEN", "pat-123")
    assert credentials_for("registry.example.org") == Credentials("<token>", "pat-123")

    monkeypatch.setenv("HUB_REGISTRY_USERNAME", "alice")
    assert credentials_for("registry.example.org") == Credentials("alice", "pat-123")


def test_github_token_for_ghcr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "gh-abc")
    monkeypatch.setenv("GITHUB_ACTOR", "djankov")
    assert credentials_for("ghcr.io") == Credentials("djankov", "gh-abc")
    assert credentials_for("other.example.org") is None  # scoped to ghcr, never sent elsewhere


def test_docker_config_auths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = base64.b64encode(b"bob:s3cret").decode()
    _write_config(tmp_path, {"auths": {"ghcr.io": {"auth": encoded}}}, monkeypatch)
    assert credentials_for("ghcr.io") == Credentials("bob", "s3cret")


def test_docker_config_username_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"auths": {"https://index.docker.io/v1/": {"username": "u", "password": "p"}}}
    _write_config(tmp_path, config, monkeypatch)
    assert credentials_for("docker.io") == Credentials("u", "p")  # the legacy long-form key


def test_registry_auth_file_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REGISTRY_AUTH_FILE is podman/skopeo's spelling of the same file."""
    path = tmp_path / "auth.json"
    encoded = base64.b64encode(b"carol:tok").decode()
    path.write_bytes(json.dumps({"auths": {"zot.example.org": {"auth": encoded}}}).encode())
    monkeypatch.setenv("REGISTRY_AUTH_FILE", str(path))
    assert credentials_for("zot.example.org") == Credentials("carol", "tok")


def test_malformed_config_degrades_to_anonymous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_bytes(b"{not json")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    assert credentials_for("ghcr.io") is None


def test_malformed_auth_entry_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, {"auths": {"ghcr.io": {"auth": "!!!not-base64!!!"}}}, monkeypatch)
    assert credentials_for("ghcr.io") is None

    _write_config(
        tmp_path,
        {"auths": {"ghcr.io": {"auth": base64.b64encode(b"nocolon").decode()}}},
        monkeypatch,
    )
    assert credentials_for("ghcr.io") is None


def test_credential_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`credHelpers`/`credsStore` shell out to docker-credential-<helper> (the standard flow)."""
    _write_config(tmp_path, {"credHelpers": {"ecr.example.org": "ecr-login"}}, monkeypatch)
    monkeypatch.setattr(
        "astro_mine.hub.registry._auth._from_helper",
        lambda helper, host: Credentials(f"{helper}-user", host),
    )
    assert credentials_for("ecr.example.org") == Credentials("ecr-login-user", "ecr.example.org")

    _write_config(tmp_path, {"credsStore": "desktop"}, monkeypatch)
    assert credentials_for("any.example.org") == Credentials("desktop-user", "any.example.org")


def test_missing_helper_binary_degrades_to_anonymous() -> None:
    assert _from_helper("definitely-not-installed", "ghcr.io") is None


def test_helper_protocol_is_the_standard_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docker-credential-<helper> get` with the host on stdin → {"Username","Secret"} on stdout."""
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["stdin"] = kwargs["input"]
        payload = json.dumps({"Username": "AWS", "Secret": "ecr-token"})
        return subprocess.CompletedProcess(argv, 0, payload, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _from_helper("ecr-login", "1234.dkr.ecr.us-east-1.amazonaws.com") == Credentials(
        "AWS", "ecr-token"
    )
    assert captured["argv"] == ["docker-credential-ecr-login", "get"]
    assert captured["stdin"] == "1234.dkr.ecr.us-east-1.amazonaws.com"


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, ""),  # the helper holds no credentials for this host
        (0, "not json"),
        (0, '{"Username": "u"}'),  # no Secret
        (0, '{"Username": "u", "Secret": ""}'),  # an empty Secret is not a credential
    ],
)
def test_helper_failures_degrade_to_anonymous(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, returncode, stdout, ""),
    )
    assert _from_helper("some-helper", "ghcr.io") is None


def test_basic_header_encoding() -> None:
    assert Credentials("u", "p").basic == base64.b64encode(b"u:p").decode()
