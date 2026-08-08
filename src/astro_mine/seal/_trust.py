"""The trust root — *whose* signature counts (conventions.md §9; hub.md §9).

Seal has been able to answer "is this signature intact and bound to this artifact?" since it
shipped. It could not answer "did someone we trust make it", and that is the question a gate
actually asks. ``verify_signature`` took a single ``trusted_public_key_pem``: omit it and **any**
key satisfied the check, which proves integrity and self-consistency but not trust; supply it and
exactly one key ever passes, which cannot be rotated without a flag day.

A trust root is therefore a **set**, not a key. That is the whole design, and every other property
falls out of it:

* **Rotation is an overlap, not a cutover.** The successor is added, both are valid for a window,
  the predecessor is removed. With one key there is an instant at which every previously signed
  artifact stops verifying, so rotation was not a procedure anyone could safely run.
* **Revocation is removal**, and because keys carry validity windows, an expiry can be *scheduled*
  rather than remembered.
* **Granularity is per key.** One org identity covers everything by default; a key may be scoped to
  particular artifact kinds when a narrower signer is wanted, without inventing a second mechanism.

**Keyed now, keyless additive later.** The keys here are the shipped `sigstore_cosign` ECDSA path,
which works offline with no account — the local tier's default and a hard requirement (CX-LOCAL).
Keyless Sigstore (Fulcio/Rekor, OIDC-bound) needs an identity provider and a network, so it cannot
be the *only* answer; hub.md §9 defers it as additive behind the same scheme, and it arrives as
another :class:`TrustedKey` variant rather than a parallel trust system.

**Where a trust root comes from** (conventions.md §9): the explicit argument wins, then
``$ASTRO_MINE_TRUST_ROOT``, then the one packaged in this wheel. A gate must be able to decide trust
on a laptop with no network, so the default has to travel with the code — a trust root that must be
fetched is a trust root the offline tier does not have.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from cryptography.hazmat.primitives import serialization

__all__ = [
    "TRUST_ROOT_ENV",
    "TrustRoot",
    "TrustRootError",
    "TrustedKey",
    "default_trust_root",
    "load_trust_root",
    "same_key",
    "trust_root_from_env",
]

#: Points at a trust-root JSON document, overriding the packaged default.
TRUST_ROOT_ENV = "ASTRO_MINE_TRUST_ROOT"

#: The packaged default, shipped inside the wheel so the offline tier has one without fetching.
_PACKAGED = "trust_root.json"


class TrustRootError(ValueError):
    """A trust-root document that cannot be read, parsed, or trusted as written."""


def _normalize(pem: bytes | str) -> bytes:
    """PEM bytes, whitespace-tidied. Storage form only; identity is :func:`same_key`'s job."""
    raw = pem.encode() if isinstance(pem, str) else pem
    return b"\n".join(line.strip() for line in raw.strip().splitlines())


def _der(pem: bytes) -> bytes | None:
    """Canonical DER for a public key, or ``None`` if it will not load."""
    try:
        key = serialization.load_pem_public_key(pem)
    except (ValueError, TypeError):
        return None
    return key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def same_key(a_pem: bytes, b_pem: bytes) -> bool:
    """Whether two PEMs are the same public key, compared by canonical DER.

    **Not a text comparison.** The same key re-wrapped, re-encoded, or written with different line
    endings is the same key, and a trust root that said otherwise would refuse a correctly signed
    artifact for a formatting difference -- a failure mode that looks exactly like an attack and is
    not one. This is the single definition of key identity in the platform; `_signing._keys_equal`
    delegates here rather than keeping a second one, on the same reasoning conventions.md §9 gives
    for a single signer implementation: two comparisons that disagree fail silently.

    Unloadable material is never equal to anything, including itself -- a malformed key cannot be
    the reason a signature is trusted.
    """
    left, right = _der(a_pem), _der(b_pem)
    return left is not None and right is not None and left == right


@dataclass(frozen=True)
class TrustedKey:
    """One signer this platform is willing to believe.

    ``identity`` is a human-readable name for *who* the key is, not a cryptographic property. It
    exists so an audit log, an error message and a rotation procedure can all refer to the same
    signer without quoting a PEM at anyone.

    ``not_before`` / ``not_after`` bound when the key is honoured. Both are optional and both are
    inclusive-open (`None` means unbounded). Scheduling an expiry is how a rotation is planned in
    advance rather than remembered on the day.

    ``kinds`` scopes the key to particular artifact kinds -- the per-kind override. ``None`` means
    every kind, which is the org-identity default; a narrower signer names the kinds it may sign.
    """

    identity: str
    public_key_pem: bytes
    not_before: datetime | None = None
    not_after: datetime | None = None
    kinds: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_key_pem", _normalize(self.public_key_pem))
        if _der(self.public_key_pem) is None:
            # A garbage key in a trust root is a misconfiguration that would otherwise present as
            # "this artifact is untrusted" on every artifact -- refuse it where it is written.
            raise TrustRootError(
                f"{self.identity!r}: public_key_pem is not a loadable public key"
            )
        if not self.identity.strip():
            raise TrustRootError(
                "a trusted key needs an identity; an anonymous root cannot be audited"
            )
        if self.not_before and self.not_after and self.not_after < self.not_before:
            raise TrustRootError(
                f"{self.identity!r}: not_after {self.not_after.isoformat()} precedes "
                f"not_before {self.not_before.isoformat()}, so the key is never valid"
            )

    def valid_at(self, moment: datetime) -> bool:
        """Whether this key is within its validity window at ``moment``."""
        if self.not_before is not None and moment < self.not_before:
            return False
        return not (self.not_after is not None and moment > self.not_after)

    def covers(self, kind: str | None) -> bool:
        """Whether this key may sign ``kind``. An unscoped key covers everything."""
        return self.kinds is None or (kind is not None and kind in self.kinds)


@dataclass(frozen=True)
class TrustRoot:
    """The set of signers a gate will accept, and the reason a rotation can be safe.

    Empty is a legitimate state and means *trust nobody* -- every signature fails. It is not the
    same as passing no trust root at all, which means *do not check identity*; conflating the two is
    how a misconfiguration turns into an open door, so they are different objects rather than
    different values of one.
    """

    keys: tuple[TrustedKey, ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, *keys: TrustedKey) -> TrustRoot:
        return cls(tuple(keys))

    @classmethod
    def single(cls, public_key_pem: bytes, *, identity: str = "unnamed") -> TrustRoot:
        """The one-key root a bare ``trusted_public_key_pem`` means.

        Kept so the old single-key callers keep their exact behaviour while the set-shaped root
        becomes the way to express anything more.
        """
        return cls((TrustedKey(identity=identity, public_key_pem=public_key_pem),))

    def accepts(
        self, signer_pem: bytes | str, *, kind: str | None = None, at: datetime | None = None
    ) -> bool:
        """Whether ``signer_pem`` is a key this root honours for ``kind`` at ``at``."""
        return self.match(signer_pem, kind=kind, at=at) is not None

    def match(
        self, signer_pem: bytes | str, *, kind: str | None = None, at: datetime | None = None
    ) -> TrustedKey | None:
        """The :class:`TrustedKey` that honours ``signer_pem``, or ``None``.

        Returns the key rather than a bool so a caller can name the signer in an audit record --
        "accepted, signed by astro-mine-release-2026" is a better log line than "accepted".
        """
        moment = at if at is not None else datetime.now(UTC)
        candidate = _normalize(signer_pem)
        for key in self.keys:
            if (
                same_key(key.public_key_pem, candidate)
                and key.valid_at(moment)
                and key.covers(kind)
            ):
                return key
        return None

    def why_rejected(
        self, signer_pem: bytes | str, *, kind: str | None = None, at: datetime | None = None
    ) -> str:
        """A specific reason ``signer_pem`` was refused -- for the error a human has to act on.

        "signing key is not trusted" sends a reader hunting; "signed by a key that expired on
        2026-03-01" sends them to the rotation procedure. Distinguishing *unknown key* from *known
        key, wrong window* from *known key, wrong kind* is the difference.
        """
        moment = at if at is not None else datetime.now(UTC)
        candidate = _normalize(signer_pem)
        known = [key for key in self.keys if same_key(key.public_key_pem, candidate)]
        if not known:
            names = ", ".join(sorted(key.identity for key in self.keys)) or "<empty trust root>"
            return f"signing key is not in the trust root (trusted: {names})"
        for key in known:
            if not key.valid_at(moment):
                window = (
                    f"{key.not_before.isoformat() if key.not_before else '-'} .. "
                    f"{key.not_after.isoformat() if key.not_after else '-'}"
                )
                return (
                    f"signed by {key.identity!r}, whose validity window ({window}) does not "
                    f"include {moment.isoformat()}"
                )
            if not key.covers(kind):
                scope = ", ".join(sorted(key.kinds or ()))
                return f"signed by {key.identity!r}, which may sign only: {scope} (not {kind!r})"
        return "signing key is not trusted"

    def identities(self) -> tuple[str, ...]:
        return tuple(key.identity for key in self.keys)


def _parse_moment(value: object, *, where: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrustRootError(f"{where}: expected an ISO-8601 string, got {type(value).__name__}")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TrustRootError(f"{where}: not an ISO-8601 timestamp: {value!r}") from exc
    # A naive timestamp compared against an aware "now" raises at the worst possible moment -- in a
    # gate, on an artifact that may be fine. Assume UTC and say so, rather than failing later.
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def load_trust_root(document: str | bytes) -> TrustRoot:
    """Parse a trust-root JSON document.

    The shape is deliberately small -- a version and a list of keys -- because it is read by a
    security gate and every field is one more thing that can be got wrong:

    .. code-block:: json

        {
          "trust_root_version": "0.1",
          "keys": [
            {"identity": "astro-mine-anchor-dev",
             "public_key_pem": "-----BEGIN PUBLIC KEY-----\\n...",
             "not_after": "2027-01-01T00:00:00Z",
             "kinds": ["policy", "world"]}
          ]
        }

    An unknown top-level field is **refused** rather than ignored, on the same reasoning as Seal's
    unknown-``require``-token rule: a typo in a security document must not quietly disable anything.
    """
    try:
        parsed = json.loads(document)
    except json.JSONDecodeError as exc:
        raise TrustRootError(f"trust root is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TrustRootError("trust root must be a JSON object")

    unknown = sorted(set(parsed) - {"trust_root_version", "keys"})
    if unknown:
        raise TrustRootError(f"unknown trust-root field(s): {', '.join(unknown)}")

    entries = parsed.get("keys", [])
    if not isinstance(entries, list):
        raise TrustRootError("trust root 'keys' must be a list")

    keys: list[TrustedKey] = []
    for index, entry in enumerate(entries):
        where = f"keys[{index}]"
        if not isinstance(entry, dict):
            raise TrustRootError(f"{where}: expected an object")
        extra = sorted(
            set(entry) - {"identity", "public_key_pem", "not_before", "not_after", "kinds"}
        )
        if extra:
            raise TrustRootError(f"{where}: unknown field(s): {', '.join(extra)}")
        pem = entry.get("public_key_pem")
        identity = entry.get("identity")
        if not isinstance(pem, str) or not isinstance(identity, str):
            raise TrustRootError(f"{where}: 'identity' and 'public_key_pem' are required strings")
        kinds = entry.get("kinds")
        if kinds is not None and not (
            isinstance(kinds, list) and all(isinstance(k, str) for k in kinds)
        ):
            raise TrustRootError(f"{where}: 'kinds' must be a list of strings")
        keys.append(
            TrustedKey(
                identity=identity,
                public_key_pem=pem.encode(),
                not_before=_parse_moment(entry.get("not_before"), where=f"{where}.not_before"),
                not_after=_parse_moment(entry.get("not_after"), where=f"{where}.not_after"),
                kinds=frozenset(kinds) if kinds is not None else None,
            )
        )
    return TrustRoot(tuple(keys))


def default_trust_root() -> TrustRoot:
    """The trust root packaged in this wheel.

    Shipped rather than fetched, because a gate has to decide trust on a laptop with no network
    (CX-LOCAL). An absent or empty packaged document yields an empty root -- *trust nobody* -- which
    fails closed: it can never be the reason an artifact is accepted.
    """
    try:
        text = resources.files("astro_mine.seal").joinpath(_PACKAGED).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - packaging accident
        return TrustRoot()
    return load_trust_root(text)


def trust_root_from_env(env: dict[str, str] | None = None) -> TrustRoot:
    """``$ASTRO_MINE_TRUST_ROOT`` (a path to a JSON document), else the packaged default."""
    source = (env if env is not None else dict(os.environ)).get(TRUST_ROOT_ENV, "").strip()
    if not source:
        return default_trust_root()
    path = Path(source).expanduser()
    try:
        return load_trust_root(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrustRootError(f"{TRUST_ROOT_ENV}={source!r}: cannot read trust root: {exc}") from exc


def resolve_trust_root(
    explicit: TrustRoot | None = None,
    *,
    trusted_public_key_pem: bytes | None = None,
    env: dict[str, str] | None = None,
) -> TrustRoot | None:
    """The precedence conventions.md §9 fixes: explicit → single key → environment → packaged.

    Returns ``None`` only when nothing at all is configured *and* the caller asked for no pinning,
    which preserves the pre-existing "any intact signature passes" behaviour for callers that have
    not opted in. A gate that decides trust must not rely on that -- it passes a root.
    """
    if explicit is not None:
        return explicit
    if trusted_public_key_pem is not None:
        return TrustRoot.single(trusted_public_key_pem)
    resolved = trust_root_from_env(env)
    return resolved if resolved.keys else None


def to_document(root: TrustRoot, *, version: str = "0.1") -> str:
    """Render ``root`` as the JSON :func:`load_trust_root` reads -- the rotation artifact."""

    def one(key: TrustedKey) -> dict[str, object]:
        out: dict[str, object] = {
            "identity": key.identity,
            "public_key_pem": key.public_key_pem.decode(),
        }
        if key.not_before is not None:
            out["not_before"] = key.not_before.isoformat()
        if key.not_after is not None:
            out["not_after"] = key.not_after.isoformat()
        if key.kinds is not None:
            out["kinds"] = sorted(key.kinds)
        return out

    payload = {"trust_root_version": version, "keys": [one(k) for k in root.keys]}
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _iter_keys(keys: Iterable[TrustedKey] | Sequence[TrustedKey]) -> tuple[TrustedKey, ...]:
    return tuple(keys)
