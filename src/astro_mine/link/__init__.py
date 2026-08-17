# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Link — the communications environment.

Models when and where agents can talk to each other and to Earth: geometric
line-of-sight + terrain occlusion (:mod:`~astro_mine.link.geometry`), relay and
ground-station contact :mod:`~astro_mine.link.windows`, a parametric link
:mod:`~astro_mine.link.budget`, the :mod:`~astro_mine.link.products`
(ContactPlan / ConnectivitySampler / CommsObservationMask) wired into the Core
Environment API, and a content-addressed :mod:`~astro_mine.link.cache`.

Link is itself a Core plugin: :mod:`~astro_mine.link.registry` declares the
``comms_model`` :class:`~astro_mine.core.registry.PluginManifest` and publishes a
contact plan to Hub as a signed, content-addressed artifact a consumer resolves by
digest; :mod:`~astro_mine.link.anchor` is the pinned comms scenario (lunar polar
relay + DSN) behind the flagship benchmark.

Phase-0 MVP scope. Geometry is ground truth; RF is a layer on top. See
``docs/architecture/link.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
