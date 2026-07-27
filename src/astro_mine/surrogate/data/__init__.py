"""Packaged data for astro-mine-surrogate — the frozen DEM training fixture (RM-P1-SURR-02).

``dem_excavation_v1.npz`` is a content-addressed particle-rollout dataset generated from the
high-fidelity SIM-06 DEM engine by ``scripts/gen_dem_dataset.py``; the learned-DEM surrogate
trains and validates against it (loaded via
:func:`astro_mine.surrogate.models.dataset.load_dem_dataset`).
"""
