"""Allocation tests."""

from __future__ import annotations

import numpy as np

from src.allocation import allocation_for_method, build_coefficients


def test_lesion_size_allocations():
    assert allocation_for_method("proposed_lsa_glqie", "small") == (2, 6)
    assert allocation_for_method("proposed_lsa_glqie", "medium") == (4, 4)
    assert allocation_for_method("proposed_lsa_glqie", "large") == (6, 2)


def test_fixed_total_coefficient_count():
    g = np.ones((3, 12))
    l = np.ones((3, 12)) * 2
    x = build_coefficients(g, l, "proposed_lsa_glqie", np.array(["small", "medium", "large"]))
    assert x.shape == (3, 8)

