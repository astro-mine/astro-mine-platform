# SPDX-License-Identifier: Apache-2.0
"""Registry credentials — the **standard** Docker/OCI mechanisms, never a bespoke scheme.

A remote pull must authenticate the way every other OCI client already does (hub.md §2 principle 6
"standards in, standards out"; §11 "ORAS + cosign ... no bespoke protocol"), so
:func:`credentials_for` resolves a registry host's credentials from the sources an operator has
*already* configured, in precedence order:

1. **Token env vars** — ``HUB_REGISTRY_TOKEN`` (with optional ``HUB_REGISTRY_USERNAME``, default
   ``<token>``, the conventional bearer-token username), and ``GITHUB_TOKEN`` for ``ghcr.io``. This
   is the CI path: a job that already has a token needs no config file.
2. **The Docker config file** — ``$DOCKER_CONFIG/config.json`` or ``~/.docker/config.json`` (also
   ``REGISTRY_AUTH_FILE``, the podman/skopeo spelling): ``auths[host].auth`` (base64
   ``user:password``), ``credHelpers[host]``, and ``credsStore``.
3. **Docker credential helpers** — ``docker-credential-<helper> get`` with the host on stdin, the
   standard helper protocol (osxkeychain, desktop, ecr-login, gcloud, …).

Nothing here is Hub-specific: `docker login`, `oras login`, or a cloud credential helper all produce
credentials this module reads. An unauthenticated (anonymous) pull is the default when no
credentials are found — public artifacts stay frictionless (hub.md §2 principle 5).
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Credentials", "credentials_for"]

#: The conventional username for a bearer/PAT credential presented over basic auth.
_TOKEN_USER = "<token>"

_HELPER_TIMEOUT_S = 10


@dataclass(frozen=True)
class Credentials:
    """A registry username/password pair (a PAT or bearer token is the password)."""

    username: str
    password: str

    @property
    def basic(self) -> str:
        """The ``Basic`` authorization value (base64 ``user:password``)."""
        raw = f"{self.username}:{self.password}".encode()
        return base64.b64encode(raw).decode("ascii")


def _config_paths() -> list[Path]:
    """Candidate Docker/OCI auth-config files, in precedence order."""
    paths: list[Path] = []
    if auth_file := os.environ.get("REGISTRY_AUTH_FILE"):
        paths.append(Path(auth_file))
    if docker_config := os.environ.get("DOCKER_CONFIG"):
        paths.append(Path(docker_config) / "config.json")
    paths.append(Path.home() / ".docker" / "config.json")
    return paths


def _read_config() -> dict[str, Any]:
    for path in _config_paths():
        try:
            config: dict[str, Any] = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue  # absent or malformed — try the next source (never fail a pull on this)
        return config
    return {}


def _decode_auth(encoded: str) -> Credentials | None:
    try:
        raw = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    username, sep, password = raw.partition(":")
    return Credentials(username, password) if sep else None


def _from_helper(helper: str, host: str) -> Credentials | None:
    """Run ``docker-credential-<helper> get`` — the standard credential-helper protocol."""
    try:
        completed = subprocess.run(
            [f"docker-credential-{helper}", "get"],
            input=host,
            capture_output=True,
            text=True,
            timeout=_HELPER_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return None
    username, secret = payload.get("Username"), payload.get("Secret")
    if not isinstance(username, str) or not isinstance(secret, str) or not secret:
        return None
    return Credentials(username, secret)


def _config_hosts(host: str) -> list[str]:
    """The keys a Docker config may file ``host`` under (Docker Hub has a legacy long form)."""
    keys = [host, f"https://{host}", f"https://{host}/v1/", f"{host}/v1/"]
    if host in ("docker.io", "index.docker.io", "registry-1.docker.io"):
        keys.append("https://index.docker.io/v1/")
    return keys


def _from_env(host: str) -> Credentials | None:
    if token := os.environ.get("HUB_REGISTRY_TOKEN"):
        return Credentials(os.environ.get("HUB_REGISTRY_USERNAME") or _TOKEN_USER, token)
    if host == "ghcr.io" and (token := os.environ.get("GITHUB_TOKEN")):
        # ghcr accepts any username with a PAT/GITHUB_TOKEN as the password.
        return Credentials(os.environ.get("GITHUB_ACTOR") or _TOKEN_USER, token)
    return None


def credentials_for(host: str) -> Credentials | None:
    """Resolve credentials for registry ``host``, or ``None`` for an anonymous (public) pull.

    Env token → Docker config ``auths`` → ``credHelpers[host]`` → ``credsStore``. Every source is
    best-effort: a missing or malformed one is skipped rather than failing the pull, so an anonymous
    pull of a public artifact always works.
    """
    if env := _from_env(host):
        return env

    config = _read_config()
    auths = config.get("auths")
    if isinstance(auths, dict):
        for key in _config_hosts(host):
            entry = auths.get(key)
            if not isinstance(entry, dict):
                continue
            if isinstance(encoded := entry.get("auth"), str) and (found := _decode_auth(encoded)):
                return found
            username, password = entry.get("username"), entry.get("password")
            if isinstance(username, str) and isinstance(password, str) and password:
                return Credentials(username, password)

    helpers = config.get("credHelpers")
    if isinstance(helpers, dict):
        for key in _config_hosts(host):
            helper = helpers.get(key)
            if isinstance(helper, str) and (found := _from_helper(helper, host)):
                return found

    if isinstance(store := config.get("credsStore"), str) and store:
        return _from_helper(store, host)
    return None
