"""
Phase-field fracture (AT-2), DOLFINx 0.9.x / 0.10.x

- 不写主时间序列 VTK
- 每 write_every 步收集到 rank0；最终写 <prefix>_fields.npz
- 快照：仅保存 d 的 PNG（rank0 用规则网格渲染）
- NPZ 中包含：
    * 原始几何：X_nodes(Nv,2), TRI(M,3), TRI_OWNER(M,)   # 仅作参考；已做去重/去退化
    * nodal: H_nodal(T,Nv), d_nodal(T,Nv)
    * cell DG0: strain_cell4(T,Nc,4), stress_cell4(T,Nc,4)   # 2x2->4, 顺序[xx, xy, yx, yy]
    * 规则网格：(可选, Ny×Nx)
         grid_fields_4d: (T, Ny, Nx, 10)
         grid_fields_2d: (T, Ny*Nx, 10)
         field_components: ["gc","ell","H","sxx","syy","sxy","exx","eyy","exy","d"]
- 可选：--npz_checkpoint_every k  -> 每收集 k 次就安全写 partial NPZ
"""

import os, tempfile
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfinx import mesh, fem, io, geometry
import dolfinx.fem.petsc as fem_petsc
import ufl

# from combined import predict
from corrector import Corrector
from .PFx_hybrid_v3_FEM import FEMGridSolver
from .PFx_hybrid_v3_utils import (
    build_problem_rect, read_mesh_xdmf, safe_savez,
    _collect_static_geometry, write_png_d_u
)

# ---------- solver ----------
def solve_phasefield(
    msh,
    E=210.0, nu=0.3, Gc=2.7e-3, ell=0.01, k_reg=1e-5,
    nsteps=100, dt=1.0, rate=5e-4,
    out_dir="/jet/home/ysunb/project/crack_ML/corrector/hybrid_test", prefix="pf",
    write_every=1,
    min_stagger=2, max_stagger=200, tol=1e-6,
    grid_nx=256, grid_ny=256,
    snapshot_every=20,
    npz_checkpoint_every=0,
    npz_compress=False,
    png_dpi=200, png_cmap="viridis", png_vmin=0.0, png_vmax=1.0,
    top_disp_value=0.1
):
    """
    相场断裂求解主函数（完全基于numpy数组，无FEM细节）
    """
    C = Corrector(predictor_path='/jet/home/ysunb/project/crack_ML/corrector/checkpoint_epoch4_ell.pth', 
                    corrector_path='/jet/home/ysunb/project/crack_ML/corrector/checkpoint_epoch2_corrector.pth',
                    nx=grid_nx, ny=grid_ny)
    comm = msh.comm
    rank = comm.rank
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()

    # 创建完全封装的FEM网格求解器
    solver = FEMGridSolver(msh, grid_nx, grid_ny, E, nu, k_reg, Gc, top_disp_value)
    
    # 在rank 0初始化网格数据
    if rank == 0:
        Ny, Nx = grid_ny, grid_nx
        d_grid = np.zeros((Ny, Nx), dtype=np.float32)
        d_pre_grid = np.zeros((Ny, Nx), dtype=np.float32)
        H_grid = np.zeros((Ny, Nx), dtype=np.float32)
        gc_grid = np.full((Ny, Nx), Gc, dtype=np.float32)
        
        # 初始化增量场
        exx_prev = np.zeros((Ny, Nx), dtype=np.float32)
        eyy_prev = np.zeros((Ny, Nx), dtype=np.float32)
        exy_prev = np.zeros((Ny, Nx), dtype=np.float32)
        sxx_prev = np.zeros((Ny, Nx), dtype=np.float32)
        syy_prev = np.zeros((Ny, Nx), dtype=np.float32)
        sxy_prev = np.zeros((Ny, Nx), dtype=np.float32)
    else:
        d_grid = d_pre_grid = H_grid = None

    time_phys = 0.0
    input_grid = []
    d_grid_history = []

    for n in range(1, nsteps+1):
        time_phys += dt

        # 保存当前d作为d_pre
        if rank == 0:
            d_pre_grid[:] = d_grid

        # 设置位移增量
        target = rate * time_phys
        solver.set_top_displacement(target)

        for it in range(1, max_stagger+1):
            # 调用完全封装的求解器（输入输出都是numpy数组）
            success, grid_fields = solver.solve_and_sample(d_grid, d_pre_grid, H_grid)
            
            if not success:
                if rank == 0:
                    print(f"[step {n}, iter {it}] FEM solve failed", flush=True)
                break
            
            # rank 0 处理网格数据和调用NN
            if rank == 0:
                # 提取网格场（都是numpy数组）
                exx_u1 = grid_fields['exx']
                eyy_u1 = grid_fields['eyy']
                exy_u1 = grid_fields['exy']
                sxx_u1 = grid_fields['sxx']
                syy_u1 = grid_fields['syy']
                sxy_u1 = grid_fields['sxy']
                H_g = grid_fields['H']
                dpreA = grid_fields['d_pre']
                
                # 计算增量
                exx_I = exx_u1 - exx_prev
                eyy_I = eyy_u1 - eyy_prev
                exy_I = exy_u1 - exy_prev
                sxx_I = sxx_u1 - sxx_prev
                syy_I = syy_u1 - syy_prev
                sxy_I = sxy_u1 - sxy_prev
                
                # 处理NaN
                pre_crack = np.isnan(dpreA).astype(np.float32)
                for arr in [H_g, dpreA, sxx_u1, syy_u1, sxy_u1, 
                            exx_u1, eyy_u1, exy_u1,
                            exx_I, eyy_I, exy_I, sxx_I, syy_I, sxy_I]:
                    arr[np.isnan(arr)] = 0.0
                
                # 构建NN输入
                input_NN = np.stack([
                    gc_grid, H_g, dpreA,
                    sxx_u1, syy_u1, sxy_u1,
                    exx_u1, eyy_u1, exy_u1,
                    pre_crack,
                    exx_I, eyy_I, exy_I,
                    sxx_I, syy_I, sxy_I
                ], axis=0).astype(np.float32)
                input_NN = np.transpose(input_NN, (0, 2, 1))
                
                # 调用NN预测
                d_NN = C.predict(
                    predictor_path='/jet/home/ysunb/project/crack_ML/corrector/checkpoint_epoch4_ell.pth', 
                    corrector_path='/jet/home/ysunb/project/crack_ML/corrector/checkpoint_epoch2_corrector.pth', 
                    npz_data=input_NN, 
                    ell=np.array([ell], dtype=np.float32)
                )
                d_grid = np.clip(d_NN.T, 0.0, 1.0).astype(np.float32)
                
                # 更新增量场
                exx_prev[:] = exx_u1
                eyy_prev[:] = eyy_u1
                exy_prev[:] = exy_u1
                sxx_prev[:] = sxx_u1
                syy_prev[:] = syy_u1
                sxy_prev[:] = sxy_u1
            
            # 目前简化：只做一次迭代
            break
        
        # 更新历史场
        H_grid = solver.update_history(H_grid)
        
        # PNG 快照
        if (snapshot_every > 0) and (n % snapshot_every == 0):     
            if rank == 0:
                png_path = write_png_d_u(out_dir, prefix, n, solver.grid_x, solver.grid_y, 
                                         d_grid, exx_u1, eyy_u1, png_dpi, png_cmap)
                print(f"[png] wrote {png_path}", flush=True)

        # 保存NPZ检查点
        if (n % npz_checkpoint_every == 0) and (npz_checkpoint_every > 0):
            if rank == 0:
                npz_path = os.path.join(out_dir, f"{prefix}_grid_data_step{n:04d}.npz")
                np.savez_compressed(
                    npz_path,
                    input_grid=np.array(input_grid),
                    d_grid=np.array(d_grid_history)
                ) if npz_compress else np.savez(
                    npz_path,
                    input_grid=np.array(input_grid),
                    d_grid=np.array(d_grid_history)
                )
                print(f"[npz] wrote checkpoint {npz_path}", flush=True)


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--nx", type=int, default=160)
    ap.add_argument("--ny", type=int, default=96)

    ap.add_argument("--xdmf", type=str, default="/ocean/projects/mat240019p/frysally/fracture/input_meshes/meshes_10.xdmf",
                    help="可选：XDMF 网格路径（设为空字符串以使用矩形网格）")
    ap.add_argument("--mesh_name", type=str, default="notched_sample",
                    help="XDMF 内 mesh 名（默认 mesh）")

    ap.add_argument("--nsteps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=5e-4)
    ap.add_argument("--Gc", type=float, default=2.7e-3)
    ap.add_argument("--ell", type=float, default=0.01)
    ap.add_argument("--E", type=float, default=210.0)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--k_reg", type=float, default=1e-5)
    ap.add_argument("--top_disp", type=float, default=0.0)

    ap.add_argument("--write_every", type=int, default=1)
    default_out = f"{Path(__file__).stem}_output"
    ap.add_argument("--out_dir", type=str, default='/jet/home/ysunb/project/crack_ML/corrector/hybrid_test')
    ap.add_argument("--prefix", type=str, default="pf")

    ap.add_argument("--grid_nx", type=int, default=256)
    ap.add_argument("--grid_ny", type=int, default=256)

    ap.add_argument("--snapshot_every", type=int, default=1,
                    help="每隔 N 步保存 d 的 PNG（0 表示关闭）")

    ap.add_argument("--npz_checkpoint_every", type=int, default=10,
                    help=">0 时每收集 k 次就安全写 partial NPZ")
    ap.add_argument("--npz_compress", action="store_true",
                    help="使用压缩保存（CPU↑, 写盘量↓）")

    ap.add_argument("--png_dpi",  type=int,   default=200)
    ap.add_argument("--png_cmap", type=str,   default="viridis")
    ap.add_argument("--png_vmin", type=float, default=0.0)
    ap.add_argument("--png_vmax", type=float, default=1.0)

    args, petsc_argv = ap.parse_known_args()

    if args.xdmf and args.xdmf.strip():
        try:
            msh = read_mesh_xdmf(args.xdmf, name=args.mesh_name)
        except Exception:
            msh = read_mesh_xdmf(args.xdmf)  # fallback to default "mesh"
    else:
        msh = build_problem_rect(L=args.L, H=args.H, nx=args.nx, ny=args.ny)

    if petsc_argv:
        PETSc.Options().insertString(" ".join(petsc_argv))

    solve_phasefield(
        msh,
        E=args.E, nu=args.nu, Gc=args.Gc, ell=args.ell, k_reg=args.k_reg,
        nsteps=args.nsteps, dt=args.dt, rate=args.rate,
        out_dir=args.out_dir, prefix=args.prefix,
        write_every=args.write_every,
        grid_nx=args.grid_nx, grid_ny=args.grid_ny,
        snapshot_every=args.snapshot_every,
        npz_checkpoint_every=args.npz_checkpoint_every,
        npz_compress=args.npz_compress,
        png_dpi=args.png_dpi, png_cmap=args.png_cmap,
        png_vmin=args.png_vmin, png_vmax=args.png_vmax,
        top_disp_value=args.top_disp
    )