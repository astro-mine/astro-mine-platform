"""Unit tests for the engines' pure-Python vector helpers (RM-P0-SIM-03)."""

from __future__ import annotations

import math

from astro_mine.sim.engines import _vecmath as vm


def test_normalize_of_zero_vector_is_zero() -> None:
    assert vm.normalize((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_normalize_yields_a_unit_vector() -> None:
    assert math.isclose(vm.norm(vm.normalize((3.0, 4.0, 0.0))), 1.0, rel_tol=1e-12)


def test_cross_is_right_handed() -> None:
    assert vm.cross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == (0.0, 0.0, 1.0)


def test_axis_angle_rotate_quarter_turn_about_z() -> None:
    x, y, z = vm.axis_angle_rotate((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2)
    assert (
        math.isclose(x, 0.0, abs_tol=1e-12)
        and math.isclose(y, 1.0)
        and math.isclose(z, 0.0, abs_tol=1e-12)
    )
