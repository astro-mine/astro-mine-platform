"""Errors raised by the Link plugin-manifest / Hub-publish path."""

from __future__ import annotations

__all__ = ["LinkRegistryError"]


class LinkRegistryError(Exception):
    """A comms-model manifest or contact-plan bundle is malformed, or Hub is unavailable.

    Raised loudly rather than degrading: a bundle that cannot be parsed, a manifest that is not
    a ``comms_model``, or a publish attempted without the optional ``astro-mine-hub`` client.
    """
