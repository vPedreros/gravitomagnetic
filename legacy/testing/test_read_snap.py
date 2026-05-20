"""
Tests for python/read_snap.py

Covers the pure helper functions that have no I/O dependencies:
- find_path: snapshot directory resolution logic
- _json_ready: numpy-type → JSON-serialisable Python type conversion
"""
import numpy as np
import pytest
from pathlib import Path

from read_snap import find_path, _json_ready


class TestFindPath:
    def test_returns_base_when_already_snapdir(self, tmp_path):
        snapdir = tmp_path / "snapdir_000"
        snapdir.mkdir()
        result = find_path(str(snapdir), 0)
        assert result == snapdir

    def test_constructs_path_from_base(self, tmp_path):
        result = find_path(str(tmp_path), 5)
        assert result == tmp_path / "snapdir_005"

    def test_zero_padded_three_digits(self, tmp_path):
        result = find_path(str(tmp_path), 42)
        assert result.name == "snapdir_042"

    def test_returns_path_object(self, tmp_path):
        result = find_path(str(tmp_path), 0)
        assert isinstance(result, Path)

    def test_non_snapdir_prefix_builds_subdir(self, tmp_path):
        # A directory that starts with "snap_" but not "snapdir_" should
        # not short-circuit; it should build snapdir_NNN inside it.
        snap_dir = tmp_path / "snap_000"
        snap_dir.mkdir()
        result = find_path(str(snap_dir), 0)
        assert result == snap_dir / "snapdir_000"


class TestJsonReady:
    def test_numpy_int64(self):
        val = _json_ready(np.int64(42))
        assert val == 42
        assert isinstance(val, int)

    def test_numpy_integer_generic(self):
        val = _json_ready(np.int32(7))
        assert val == 7
        assert isinstance(val, int)

    def test_numpy_float64(self):
        val = _json_ready(np.float64(3.14))
        assert val == pytest.approx(3.14)
        assert isinstance(val, float)

    def test_numpy_floating_generic(self):
        val = _json_ready(np.float32(1.5))
        assert isinstance(val, float)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        val = _json_ready(arr)
        assert val == [1, 2, 3]
        assert isinstance(val, list)

    def test_plain_string_passthrough(self):
        val = _json_ready("hello")
        assert val == "hello"

    def test_plain_int_passthrough(self):
        val = _json_ready(10)
        assert val == 10

    def test_none_passthrough(self):
        val = _json_ready(None)
        assert val is None
