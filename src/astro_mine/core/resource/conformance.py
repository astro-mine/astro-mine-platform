"""Resource-field API contract-test utility (prospect.md §2/§3).

The consumer-driven conformance check a Prospect field backend runs in its own CI to prove
it honors the Core :class:`~astro_mine.core.resource.protocol.ResourceField` contract —
analogous to :func:`astro_mine.core.env.check_environment`. It drives the query surface at a
sample point and asserts the contract: the object satisfies the Protocol, the metadata
properties are present, every scalar query returns a ``float``, ``sample`` returns the
requested number of draws, and ``posterior`` returns an uncertainty-first
:class:`~astro_mine.core.resource.model.FieldDistribution` pairing a mean with a variance.
Raises :class:`ResourceFieldContractError` on any violation.
"""

from __future__ import annotations

from astro_mine.core.resource.model import FieldDistribution, Position
from astro_mine.core.resource.protocol import ResourceField

__all__ = ["ResourceFieldContractError", "check_resource_field"]


class ResourceFieldContractError(AssertionError):
    """Raised when a field violates the Core ResourceField API contract."""


def check_resource_field(field: ResourceField, *, position: Position = (0.0, 0.0, 0.0)) -> None:
    """Assert ``field`` honors the Core ResourceField API v0.1 contract.

    Drives the metadata properties and the ``mean``/``variance``/``quantile``/``sample``/
    ``posterior`` query surface at ``position`` and checks the return shapes — in particular
    that ``posterior`` yields an uncertainty-first
    :class:`~astro_mine.core.resource.model.FieldDistribution`. Returns ``None`` on success.
    """
    if not isinstance(field, ResourceField):
        raise ResourceFieldContractError(
            "object does not satisfy the ResourceField protocol (missing species/unit/frame/"
            "mean/variance/quantile/sample/posterior)"
        )

    for name in ("species", "unit"):
        if not isinstance(getattr(field, name), str):
            raise ResourceFieldContractError(f"{name} must be a str")

    for name in ("mean", "variance"):
        value = getattr(field, name)(position)
        if not isinstance(value, float):
            raise ResourceFieldContractError(
                f"{name}() must return a float, got {type(value).__name__}"
            )

    if not isinstance(field.quantile(position, 0.5), float):
        raise ResourceFieldContractError("quantile() must return a float")

    draws = field.sample(position, n=3, seed=0)
    if not isinstance(draws, tuple) or len(draws) != 3:
        raise ResourceFieldContractError("sample(n=3) must return a 3-tuple of draws")

    dist = field.posterior(position)
    if not isinstance(dist, FieldDistribution):
        raise ResourceFieldContractError(
            f"posterior() must return a FieldDistribution, got {type(dist).__name__}"
        )
    if not isinstance(dist.mean, float) or not isinstance(dist.variance, float):
        raise ResourceFieldContractError(
            "posterior() must pair a float mean with a float variance (uncertainty-first)"
        )
