# SPDX-License-Identifier: Apache-2.0
"""The ``astro-mine-hub`` client SDK + CLI (RM-P1-HUB-06).

Resolve, verify, pull, cache, and publish artifacts against **any** OCI registry with **no hosted
Hub** — the tier-1 path that must always work (hub.md principle 7, §7). The client **re-verifies at
pull, before Core loads the plugin** (hub.md §2.3; ``LUNAR-SR-002``), failing closed on tampered
bytes, and serves pulls from a content-addressed cache. [Bench](bench.md) resolves submissions by
digest through this client — the other half of the academic flywheel.

Backlog: RM-P1-HUB-06 — astro-mine-hub#6
"""

from __future__ import annotations

from astro_mine.hub.client._client import HubClient, PayloadLayer, catalog_from_registry

__all__ = ["HubClient", "PayloadLayer", "catalog_from_registry"]
