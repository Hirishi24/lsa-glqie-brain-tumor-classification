"""ROI tests."""

from __future__ import annotations

import numpy as np

from src.roi_processing import expand_bbox, local_crop, tumor_bbox


def test_tumour_bbox_correctness():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:8] = 1
    assert tumor_bbox(mask) == (2, 5, 3, 8)


def test_bbox_clipping_and_local_crop():
    image = np.ones((20, 20), dtype=np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    assert expand_bbox(tumor_bbox(mask), mask.shape, 1.0) == (0, 4, 0, 4)
    crop = local_crop(image, mask, 16, 0.15)
    assert crop.shape == (16, 16)

