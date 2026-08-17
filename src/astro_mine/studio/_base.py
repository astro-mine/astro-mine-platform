# SPDX-License-Identifier: Apache-2.0
"""Shared Pydantic base for every Studio-owned artifact.

Studio adds **no Core surface** (studio.md §2, §12): these are Studio-local models
that *compose* Core schemas (``ObjectiveSpec``, SADF, the plugin manifest) and
reference sibling-produced bytes by content hash — never a private wire contract on
the narrow waist. ``extra="forbid"`` mirrors Core's fail-loud stance: an unknown or
typo'd field is rejected at construction, not silently carried.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StudioModel(BaseModel):
    """Base for mutable Studio models (drafts, requests)."""

    model_config = ConfigDict(extra="forbid")


class FrozenStudioModel(StudioModel):
    """Base for **produced, content-addressed** artifacts.

    Once a study runs or a campaign is handed off, the artifact is frozen and
    content-addressed (studio.md §5 lifecycle); immutability makes the content hash a
    stable identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
