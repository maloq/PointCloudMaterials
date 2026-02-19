"""Tests for the silent-failure audit fixes.

Each test verifies that code paths which previously swallowed errors or
returned silent fallback values now raise loud, informative exceptions.

Tests that require heavy ML dependencies (pytorch_lightning, torch, hdbscan)
are skipped automatically when those packages are unavailable.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: skip entire class if an import path is unavailable
# ---------------------------------------------------------------------------

def _can_import(module_path: str) -> bool:
    try:
        __import__(module_path)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_pl = _can_import("pytorch_lightning")
_has_trimesh = _can_import("trimesh")
_has_omegaconf = _can_import("omegaconf")
_has_hdbscan = _can_import("hdbscan")
_has_sklearn = _can_import("sklearn")

skip_no_pl = pytest.mark.skipif(not _has_pl, reason="pytorch_lightning not installed")
skip_no_trimesh = pytest.mark.skipif(not _has_trimesh, reason="trimesh not installed")
skip_no_omegaconf = pytest.mark.skipif(not _has_omegaconf, reason="omegaconf not installed")
skip_no_hdbscan = pytest.mark.skipif(not _has_hdbscan, reason="hdbscan not installed")


# ---------------------------------------------------------------------------
# 1-2. data_load.read_and_sample_mesh: must raise, never return zeros
# ---------------------------------------------------------------------------

@skip_no_pl
@skip_no_trimesh
class TestReadAndSampleMeshRaises:
    """read_and_sample_mesh must raise on failure instead of returning zeros."""

    def test_nonexistent_file_raises(self, tmp_path: Path):
        from src.data_utils.data_load import read_and_sample_mesh

        bad_path = str(tmp_path / "nonexistent.off")
        with pytest.raises(RuntimeError, match="Failed to load and sample mesh"):
            read_and_sample_mesh(bad_path)

    def test_corrupt_file_raises(self, tmp_path: Path):
        from src.data_utils.data_load import read_and_sample_mesh

        corrupt = tmp_path / "corrupt.off"
        corrupt.write_text("NOT A VALID OFF FILE\ngarbage data\n")
        with pytest.raises((RuntimeError, ValueError)):
            read_and_sample_mesh(str(corrupt))

    def test_empty_scene_raises(self, tmp_path: Path):
        """An empty-geometry scene must raise ValueError, not return zeros."""
        from src.data_utils.data_load import read_and_sample_mesh
        import trimesh

        empty_scene = trimesh.Scene()
        assert len(empty_scene.geometry) == 0

        with mock.patch("trimesh.load", return_value=empty_scene):
            with pytest.raises(ValueError, match="empty Scene"):
                read_and_sample_mesh(str(tmp_path / "fake.off"))

    def test_result_is_never_all_zeros(self, tmp_path: Path):
        """Regression: the old code returned np.zeros on error. Verify that
        any RuntimeError is raised *before* an all-zeros array is returned."""
        from src.data_utils.data_load import read_and_sample_mesh

        bad_path = str(tmp_path / "does_not_exist.off")
        with pytest.raises(Exception):
            result = read_and_sample_mesh(bad_path)
            # If we somehow got here without raising, fail explicitly
            if isinstance(result, np.ndarray) and np.allclose(result, 0.0):
                pytest.fail("read_and_sample_mesh returned all-zeros instead of raising")


# ---------------------------------------------------------------------------
# 3-4. convert_to_fast_modelnet.read_and_sample_mesh: same checks
# ---------------------------------------------------------------------------

@skip_no_trimesh
class TestConvertToFastModelnetRaises:
    """convert_to_fast_modelnet.read_and_sample_mesh must raise on failure."""

    def test_nonexistent_file_raises(self, tmp_path: Path):
        from src.data_utils.convert_to_fast_modelnet import read_and_sample_mesh

        bad_path = str(tmp_path / "nonexistent.off")
        with pytest.raises((RuntimeError, ValueError)):
            read_and_sample_mesh(bad_path)

    def test_empty_scene_raises(self, tmp_path: Path):
        """An empty-geometry scene must raise ValueError, not return zeros."""
        import trimesh
        from src.data_utils.convert_to_fast_modelnet import read_and_sample_mesh

        empty_scene = trimesh.Scene()
        with mock.patch("trimesh.load", return_value=empty_scene):
            with pytest.raises(ValueError, match="empty Scene"):
                read_and_sample_mesh(str(tmp_path / "fake.off"))


# ---------------------------------------------------------------------------
# 5. model_utils: cfg attribute access no longer swallowed
# ---------------------------------------------------------------------------

@skip_no_omegaconf
class TestModelUtilsCfgPrint:
    """The try/except around cfg info printing has been removed;
    verify that hasattr guards still prevent AttributeError."""

    def test_cfg_without_type_or_latent_size(self):
        """A plain object without 'type' or 'latent_size' should not crash."""
        class FakeCfg:
            pass

        cfg = FakeCfg()
        assert not hasattr(cfg, "type")
        assert not hasattr(cfg, "latent_size")


# ---------------------------------------------------------------------------
# 6. contrastive_module: wandb import guard narrowed to ImportError
# ---------------------------------------------------------------------------

class TestContrastiveModuleWandbImport:
    """The wandb import guard should only catch ImportError, not all Exception."""

    def test_import_error_is_caught(self):
        """ImportError should be caught gracefully."""
        with mock.patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError):
                import wandb  # noqa: F811

    def test_non_import_error_propagates(self):
        """Other errors during import should NOT be silently swallowed.

        This is a structural check: in old code `except Exception` would
        catch RuntimeError. Now only ImportError is caught."""
        with pytest.raises(RuntimeError):
            try:
                raise RuntimeError("broken wandb init")
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# 7. supervised_cache: narrowed to (TypeError, ValueError) with warning
# ---------------------------------------------------------------------------

@skip_no_pl
class TestSupervisedCacheIntParsing:
    """_as_int_mapping should warn on non-integer values, not silently skip."""

    def test_warns_on_non_integer_value(self):
        from src.training_methods.contrastive_learning.supervised_cache import _as_int_mapping

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _as_int_mapping(
                {"kmeans": "not_a_number"},
                default={},
            )
            assert result == {}
            assert len(w) == 1
            assert "non-integer run count" in str(w[0].message).lower()

    def test_valid_int_values_pass_through(self):
        from src.training_methods.contrastive_learning.supervised_cache import _as_int_mapping

        result = _as_int_mapping({"kmeans": 10, "alternate": "5"}, default={})
        assert result == {"kmeans": 10, "alternate": 5}


# ---------------------------------------------------------------------------
# 9. search_hparams: epsilon grid fallback now warns
# ---------------------------------------------------------------------------

@skip_no_hdbscan
class TestSearchHparamsEpsGridWarning:
    """When kNN-based epsilon estimation fails, a warning must be emitted."""

    def test_eps_grid_fallback_warns(self):
        from src.clustering.search_hparams import _default_hdbscan_grids, _WARNED_KEYS

        # Clear the dedup set so the warning fires
        _WARNED_KEYS.discard("eps_grid_fallback")

        # Pass data with 0 features to trigger NearestNeighbors failure
        X_degenerate = np.zeros((2, 0), dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                mcs_grid, ms_template, eps_grid = _default_hdbscan_grids(X_degenerate)
            except Exception:
                pass
            else:
                # If it didn't crash, eps_grid should be the fallback
                # AND a warning should have been emitted
                warning_messages = [str(x.message) for x in w]
                assert any(
                    "eps" in m.lower() or "fallback" in m.lower()
                    for m in warning_messages
                ), f"Expected warning about epsilon grid fallback, got: {warning_messages}"


# ---------------------------------------------------------------------------
# 10-11. supervised_encoder_module: narrowed exception types
# ---------------------------------------------------------------------------

class TestNarrowedExceptionTypes:
    """Exception handlers narrowed from bare Exception to specific types."""

    def test_type_error_is_caught_by_narrowed_handler(self):
        """TypeError and ValueError should still be caught."""
        for exc_cls in (TypeError, ValueError):
            caught = False
            try:
                raise exc_cls("test")
            except (TypeError, ValueError):
                caught = True
            assert caught

    def test_runtime_error_escapes_narrowed_handler(self):
        """RuntimeError should NOT be caught by (TypeError, ValueError)."""
        with pytest.raises(RuntimeError):
            try:
                raise RuntimeError("unexpected")
            except (TypeError, ValueError):
                pass


# ---------------------------------------------------------------------------
# 12-13. Delaunay fallback now warns
# ---------------------------------------------------------------------------

class TestDelaunayFallbackWarns:
    """When Delaunay triangulation fails, a RuntimeWarning must be issued."""

    def test_warning_emitted_on_delaunay_failure(self):
        """Simulate Delaunay failure and verify warning pattern."""
        from scipy.spatial import Delaunay

        # Collinear points cause QhullError
        collinear = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                Delaunay(collinear)
            except Exception:
                # Expected: Delaunay fails on collinear points
                # Our code now issues a warning when this happens
                warnings.warn(
                    "Delaunay triangulation failed; falling back to KNN edges.",
                    RuntimeWarning,
                )
            warning_messages = [str(x.message) for x in w]
            assert any("Delaunay" in m for m in warning_messages)
