"""Adapter around the phase-field FEM solver in `PFx_hybrid_v3.py`.

This module deliberately does not rewrite the solver. It provides a small
interface the hybrid pipeline can call:

    solve(fields) -> dict of solution fields

When the real FEM stack (dolfinx) is unavailable, `solve` raises
`FemUnavailableError` so the hybrid pipeline can fall back to a pure
neural forward pass or fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class FemUnavailableError(RuntimeError):
    pass


def is_fem_available() -> bool:
    """Probe for an importable FEM stack. Does NOT mean solve() is implemented."""
    try:
        import dolfinx  # noqa: F401
    except Exception:
        return False
    return True


def is_fem_solve_implemented() -> bool:
    """True only when `solve()` is Phase-3-ready. Currently unconditionally False."""
    return False


@dataclass
class FemConfig:
    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    ell: float = 0.01
    k_reg: float = 1e-5
    nsteps: int = 100
    dt: float = 1.0
    rate: float = 5e-4
    grid_nx: int = 256
    grid_ny: int = 256
    max_stagger: int = 200
    min_stagger: int = 2
    tol: float = 1e-6
    top_disp_value: float = 0.1

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "FemConfig":
        """Read FEM settings. If `solver.config` points at a solver YAML,
        merge its material/solver/mesh blocks in first (inline keys win).

        Inline overrides must nest under `material:`, `solver:`, or `mesh:`
        sub-dicts — e.g. `solver: {config: X, solver: {nsteps: 10}}`. Flat
        scalar keys directly under `solver:` (other than `config`) are
        rejected to avoid silently dropping user-supplied values.
        """
        s = dict(cfg.get("solver") or {})
        ref = s.pop("config", None)
        unknown = {k: v for k, v in s.items() if k not in {"material", "solver", "mesh"}}
        if unknown:
            raise ValueError(
                f"FemConfig.from_cfg: unexpected flat key(s) under `solver:`: "
                f"{sorted(unknown)!r}. Nest them under material/solver/mesh."
            )
        if ref:
            try:
                from ..utils.config import load_yaml, REPO_ROOT
            except ImportError:
                from utils.config import load_yaml, REPO_ROOT  # type: ignore
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                ref_path = REPO_ROOT / ref_path
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"solver.config points at {ref_path} which does not exist"
                )
            base = load_yaml(ref_path)
            for key in ("material", "solver", "mesh"):
                merged = dict(base.get(key) or {})
                merged.update(s.get(key) or {})
                s[key] = merged
        mat = s.get("material") or {}
        num = s.get("solver") or {}
        mesh = s.get("mesh") or {}
        return cls(
            E=float(mat.get("E", cls.E)),
            nu=float(mat.get("nu", cls.nu)),
            Gc=float(mat.get("Gc", cls.Gc)),
            ell=float(mat.get("length_scale", cls.ell)),
            k_reg=float(mat.get("k_reg", cls.k_reg)),
            nsteps=int(num.get("nsteps", cls.nsteps)),
            dt=float(num.get("dt", cls.dt)),
            rate=float(num.get("rate", cls.rate)),
            max_stagger=int(num.get("nonlinear_max_iters", cls.max_stagger)),
            min_stagger=int(num.get("nonlinear_min_iters", cls.min_stagger)),
            tol=float(num.get("nonlinear_tol", cls.tol)),
            grid_nx=int(mesh.get("grid_nx", cls.grid_nx)),
            grid_ny=int(mesh.get("grid_ny", cls.grid_ny)),
            top_disp_value=float(num.get("top_disp_value", cls.top_disp_value)),
        )


def solve(
    fields: Mapping[str, np.ndarray],
    fem_cfg: FemConfig,
    max_iters: int = 1,
) -> dict[str, np.ndarray]:
    """Run the phase-field solver for one prediction step.

    `fields` should include at least a predicted parameter field (`Gc`, or
    an initial guess for `d`) and must be numpy arrays on a regular grid.

    Returns at minimum:
        d:       [Ny, Nx] damage field
        u:       [Ny, Nx, 2] displacement (optional; may be absent)
        stress:  [Ny, Nx, 3] (sxx, syy, sxy) (optional)
    """
    try:
        import dolfinx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise FemUnavailableError(
            "dolfinx is not importable. The FEM adapter is a placeholder "
            "until the fenicsx-env is active."
        ) from e

    # Intentionally unimplemented: the production solver in
    # src/fem/PFx_hybrid_v3.py runs multi-step transients and depends on a
    # `Corrector` module not checked into this repo. Wiring that up to
    # per-sample batch inference is a Phase-3 task; the hybrid training code
    # calls this function through `HybridPredictor.step`, which currently
    # falls back to the neural prediction (see src/hybrid/paramfem.py).
    raise FemUnavailableError(
        "solve() is a Phase-3 placeholder. Use HybridPredictor(no_fem=True) "
        "until the differentiable FEM adapter is wired in."
    )
