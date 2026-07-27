"""Shared Kubernetes-manifest helpers -- dep-free dicts, YAML behind ``[cluster]``.

Cloud generates Kubernetes objects (Jobs, Argo Workflows, RayJobs, quotas, network
policies, admission policies) from typed contracts. Those objects are the canonical,
**unit-testable** form here: plain ``dict`` manifests built by the ``engines/``, ``sched/``,
``autoscale/``, ``gpu/`` and ``tenancy/`` modules (``cloud.md`` §3). Rendering a manifest to
YAML or applying it needs heavier machinery (``pyyaml`` / the ``kubernetes`` client) that
lives behind the ``[cluster]`` extra, so the sacred dependency-free local tier never pulls
it in (``cloud.md`` §2 principle 2; ``conventions.md`` §7).

This module owns the cross-cutting bits every generator shares: the platform label schema
(so every object is attributable to a tenant / run / component), RFC-1123 name
sanitisation, and the pure-Python env/metadata builders.

Backlog: RM-P1-CLOUD-01 -- https://github.com/astro-mine/astro-mine-cloud/issues/12
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "ENV_JOBSPEC",
    "LABEL_COMPONENT",
    "LABEL_MANAGED_BY",
    "LABEL_PART_OF",
    "LABEL_QUEUE_NAME",
    "LABEL_RUN",
    "LABEL_TENANT",
    "MANAGED_BY",
    "Manifest",
    "env_var_list",
    "labels",
    "object_meta",
    "sanitize_name",
    "to_yaml",
]

#: Canonical manifest form: a plain JSON-serialisable mapping. Kubernetes accepts JSON as
#: well as YAML, so the dict *is* the manifest; :func:`to_yaml` is a convenience.
Manifest = dict[str, Any]

#: The env var carrying a run's :class:`~astro_mine.cloud.submission.jobspec.JobSpec` (as JSON)
#: into its container. This is the contract between the *compile* side (``engines/``, which
#: writes it into the container env) and the *runtime* side
#: (:mod:`astro_mine.cloud.submission.harness`, the in-pod entrypoint, which reads it back).
#:
#: It lives here, in the dep-free manifest module, because ``engines`` must not import
#: ``submission`` -- ``submission`` imports ``engines`` to compile, so the reverse would cycle.
#: An env var rather than a ConfigMap keeps the pod free of any RBAC to read its own Job back
#: from the API server and keeps the compiled object self-contained.
ENV_JOBSPEC = "ASTRO_MINE_JOBSPEC"

#: ``app.kubernetes.io/managed-by`` value stamped on every object Cloud emits.
MANAGED_BY = "astro-mine-cloud"

LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_PART_OF = "app.kubernetes.io/part-of"
LABEL_COMPONENT = "app.kubernetes.io/component"
#: Platform-owned label domain (mirrors the ``astro-mine.org`` convention used in manifests).
LABEL_TENANT = "astro-mine.org/tenant"
LABEL_RUN = "astro-mine.org/run"
#: Kueue's own label: the ``LocalQueue`` an object is admitted through. Kueue is the queueing
#: authority, so the key is *its* domain, not ours (``cloud.md`` §4).
LABEL_QUEUE_NAME = "kueue.x-k8s.io/queue-name"

_RFC1123_MAX = 63
_RFC1123_INVALID = re.compile(r"[^a-z0-9-]+")
_RFC1123_EDGE = re.compile(r"^-+|-+$")


def sanitize_name(name: str) -> str:
    """Coerce *name* into a valid RFC-1123 label (lowercase alnum + ``-``, <=63 chars).

    Kubernetes object names are DNS-1123 labels; a tenant, sweep, or step name that came
    from a human must be normalised before it can go into ``metadata.name``. Empty results
    are rejected loudly rather than producing an unnameable object.
    """
    slug = _RFC1123_EDGE.sub("", _RFC1123_INVALID.sub("-", name.strip().lower()))[:_RFC1123_MAX]
    slug = _RFC1123_EDGE.sub("", slug)
    if not slug:
        raise ValueError(f"name {name!r} has no RFC-1123-safe characters")
    return slug


def labels(
    *,
    component: str,
    tenant: str | None = None,
    run: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the standard label set: managed-by / part-of / component (+ tenant / run).

    Every Cloud-emitted object carries these so it is attributable and selectable -- a
    default-deny NetworkPolicy can select a tenant's pods, and cost accounting can group by
    run (``cloud.md`` §9, §10).
    """
    out: dict[str, str] = {
        LABEL_MANAGED_BY: MANAGED_BY,
        LABEL_PART_OF: "astro-mine",
        LABEL_COMPONENT: component,
    }
    if tenant is not None:
        out[LABEL_TENANT] = sanitize_name(tenant)
    if run is not None:
        out[LABEL_RUN] = sanitize_name(run)
    if extra:
        out.update(extra)
    return out


def object_meta(
    name: str,
    *,
    namespace: str | None = None,
    tenant: str | None = None,
    run: str | None = None,
    component: str,
    annotations: Mapping[str, str] | None = None,
    extra_labels: Mapping[str, str] | None = None,
) -> Manifest:
    """Build an ``ObjectMeta`` with sanitised name + the standard label set."""
    meta: Manifest = {
        "name": sanitize_name(name),
        "labels": labels(component=component, tenant=tenant, run=run, extra=extra_labels),
    }
    if namespace is not None:
        meta["namespace"] = sanitize_name(namespace)
    if annotations:
        meta["annotations"] = dict(annotations)
    return meta


def env_var_list(env: Mapping[str, str]) -> list[Manifest]:
    """Render an env mapping as a K8s container ``env`` list, sorted for determinism.

    A dict is order-insensitive but a rendered manifest must be byte-stable so golden-run
    determinism gates hold (``cloud.md`` §10); sorting the keys guarantees it.
    """
    return [{"name": key, "value": value} for key, value in sorted(env.items())]


def to_yaml(manifests: Manifest | Iterable[Manifest]) -> str:
    """Render one manifest (or a stream) to multi-document YAML.

    Requires ``pyyaml`` (the ``[cluster]`` extra) -- rendering/applying is a cluster
    concern, so the dependency-free tier that only *builds* dict manifests never imports it.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without [cluster]
        raise ModuleNotFoundError(
            "to_yaml() needs pyyaml; install the 'cluster' extra: pip install "
            "'astro-mine-cloud[cluster]'"
        ) from exc
    docs = [manifests] if isinstance(manifests, dict) else list(manifests)
    rendered: str = yaml.safe_dump_all(docs, sort_keys=True, default_flow_style=False)
    return rendered
