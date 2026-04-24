"""Backward-compatible re-export.

Canonical location is `src/data/dataset.py`. Existing scripts that import
`utils.dataset.ChunkedScalarDatasetEfficient` continue to work.
"""
try:
    # When `src/` is a package (new entrypoints)
    from ..data.dataset import ChunkedScalarDatasetEfficient  # type: ignore
except ImportError:
    # When `src/` is on sys.path and `utils`/`data` are top-level
    from data.dataset import ChunkedScalarDatasetEfficient  # type: ignore

__all__ = ["ChunkedScalarDatasetEfficient"]
