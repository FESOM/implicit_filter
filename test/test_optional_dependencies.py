"""
pandas and scikit-learn are only needed by the NEMO north-fold helper, so they
are optional. When they are missing the failure must name the extra that
provides them rather than surfacing a bare ModuleNotFoundError.
"""
import sys
from unittest.mock import patch

import pytest

from implicit_filter.utils._auxiliary import find_adjacent_points_north


BLOCKED = {
    "sklearn": None,
    "sklearn.linear_model": None,
}


class TestMissingOptionalDependency:
    def test_error_names_the_extra(self):
        with patch.dict(sys.modules, BLOCKED):
            with pytest.raises(ImportError, match=r"implicit_filter\[nemo\]"):
                find_adjacent_points_north(None, 1.0)

    def test_error_names_the_missing_packages(self):
        with patch.dict(sys.modules, BLOCKED):
            with pytest.raises(ImportError, match="scikit-learn"):
                find_adjacent_points_north(None, 1.0)

    def test_error_explains_why_it_is_needed(self):
        with patch.dict(sys.modules, BLOCKED):
            with pytest.raises(ImportError, match="north"):
                find_adjacent_points_north(None, 1.0)


class TestPresentDependency:
    def test_no_error_when_available(self):
        """With sklearn installed the guard must not fire.

        Passing ds_mm=None then fails later for an unrelated reason; the point
        is only that it is not an ImportError about optional dependencies.
        """
        pytest.importorskip("sklearn")
        with pytest.raises(Exception) as excinfo:
            find_adjacent_points_north(None, 1.0)
        assert "implicit_filter[nemo]" not in str(excinfo.value)
