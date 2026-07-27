"""Namespace tiers + governance actions — verified publishers, yank, deprecate (RM-P1-HUB-05).

Tiered namespaces (hub.md §9): **open** (community, signed, unreviewed) vs **curated/verified**
(reviewed, verified-publisher). The trust tier is a queryable facet a consumer or [Bench](bench.md)
can require ("verified-publisher + valid provenance"). Yank and deprecation are **auditable
governance actions** (hub.md §5, §9): each mutates the entry's facet, persists it through the
catalog (so it holds on either backend), and writes an :class:`~astro_mine.hub.policy.AuditRecord`.

Backlog: RM-P1-HUB-05 — https://github.com/astro-mine/astro-mine-hub/issues/5
"""

from __future__ import annotations

from astro_mine.hub.index import Catalog, CatalogEntry
from astro_mine.hub.policy import AuditLog, AuditRecord
from astro_mine.hub.registry import RegistryClient
from astro_mine.hub.supply_chain import SupplyChainError, verify_admissible

__all__ = ["NAMESPACES", "CurationError", "deprecate", "promote", "yank"]

#: The namespace trust tiers (a queryable facet).
NAMESPACES: tuple[str, ...] = ("open", "curated", "verified")


class CurationError(Exception):
    """A governance action targets an unknown artifact or namespace."""


def _entry(catalog: Catalog, reference: str) -> CatalogEntry:
    entry = catalog.get(reference)
    if entry is None:
        raise CurationError(f"unknown artifact {reference!r}")
    return entry


def _audit(audit: AuditLog | None, action: str, reference: str, reason: str) -> None:
    if audit is not None:
        audit.record(AuditRecord(action=action, reference=reference, allowed=True, reason=reason))


def promote(
    catalog: Catalog,
    reference: str,
    *,
    to: str = "curated",
    audit: AuditLog | None = None,
    registry: RegistryClient | None = None,
    trusted_public_key_pem: bytes | None = None,
) -> CatalogEntry:
    """Promote an artifact into the ``to`` namespace tier (default ``curated``); audited.

    Promotion into a **trusted** tier (anything above ``open``) re-verifies the artifact's evidence
    first — signature, SLSA provenance, SBOM — and refuses otherwise, writing an audited denial.
    This is ``hub.md`` §2 principle 3's *"unsigned content is never promoted to a verified
    namespace"*, which until now described a check that did not exist: the only gate was that the
    tier's **name** was spelled correctly.

    ``registry`` is required to promote above ``open``, because evidence lives in the registry and
    a promotion that cannot check it cannot be fail-closed. Demoting back to ``open`` needs no
    registry — lowering trust is always safe.
    """
    if to not in NAMESPACES:
        raise CurationError(f"unknown namespace {to!r}; expected one of {NAMESPACES}")
    entry = _entry(catalog, reference)
    if to != "open":
        if registry is None:
            _audit(audit, "promote", reference, f"denied: no registry to verify {to!r} promotion")
            raise CurationError(
                f"promoting {reference!r} to {to!r} requires a registry to verify its evidence "
                f"against; promotion into a trusted tier cannot be granted unverified"
            )
        try:
            # The same gate admission runs — re-checked here rather than trusted from publish
            # time, because promotion *grants* trust and the registry may have changed since.
            verify_admissible(
                registry,
                entry.manifest,
                digest=entry.digest,
                trusted_public_key_pem=trusted_public_key_pem,
            )
        except SupplyChainError as exc:
            _audit(audit, "promote", reference, f"denied: {exc}")
            raise CurationError(
                f"cannot promote {reference!r} to {to!r}: its supply-chain evidence does not "
                f"verify ({exc})"
            ) from exc
    entry.namespace = to
    catalog.add(entry)  # persist through the backend (idempotent replace)
    _audit(audit, "promote", reference, f"promoted to {to}")
    return entry


def yank(
    catalog: Catalog, reference: str, *, reason: str = "yanked", audit: AuditLog | None = None
) -> CatalogEntry:
    """Yank an artifact — resolution refuses it by default; bytes stay pullable by digest."""
    entry = _entry(catalog, reference)
    entry.yanked = True
    catalog.add(entry)
    _audit(audit, "yank", reference, reason)
    return entry


def deprecate(
    catalog: Catalog, reference: str, *, reason: str = "deprecated", audit: AuditLog | None = None
) -> CatalogEntry:
    """Deprecate an artifact — still pullable, flagged in discovery. Audited."""
    entry = _entry(catalog, reference)
    entry.deprecated = True
    catalog.add(entry)
    _audit(audit, "deprecate", reference, reason)
    return entry
