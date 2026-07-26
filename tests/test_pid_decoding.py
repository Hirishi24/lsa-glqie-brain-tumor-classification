"""PID decoding tests."""

from __future__ import annotations

import numpy as np

from src.dataset_io import decode_matlab_char


def test_decode_uint16_pid():
    arr = np.asarray([ord(c) for c in "MR049453B"], dtype=np.uint16).reshape(-1, 1)
    assert decode_matlab_char(arr) == "MR049453B"


def test_decode_removes_nulls_and_whitespace():
    arr = np.asarray([32, 65, 66, 0, 67, 32], dtype=np.uint16).reshape(-1, 1)
    assert decode_matlab_char(arr) == "ABC"

