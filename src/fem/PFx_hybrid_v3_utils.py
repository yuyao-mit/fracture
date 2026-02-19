"""
工具函数模块 - 相场断裂模拟

包含网格处理、几何收集、数据采样等工具函数
"""

import os
import tempfile
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- 网格创建与读取 ----------

def build_problem_rect(L=1.0, H=0.5, nx=160, ny=96, use_quads=False):
    """创建矩形网格"""
    cell = mesh.CellType.quadrilateral if use_quads else mesh.CellType.triangle
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, -H]), np.array([L, H])],
        [nx, ny],
        cell_type=cell,
    )


def read_mesh_xdmf(path, name="mesh"):
    """从XDMF文件读取网格"""
    with io.XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
        msh = xdmf.read_mesh(name=name)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, 0)
    return msh


# ---------- 范数与收敛性检查 ----------

def l2_norm_sq(msh, expr):
    """计算L2范数的平方"""
    return float(fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx)))


def rel_change(msh, f, f_old, eps=1e-14):
    """计算相对变化"""
    num = l2_norm_sq(msh, f - f_old)
    den = l2_norm_sq(msh, f_old) + eps
    return (num / den) ** 0.5


# ---------- 文件I/O ----------

def safe_savez(final_path, do_fsync=True, compressed=False, **arrays):
    """安全地保存NPZ文件（原子操作）"""
    d = os.path.dirname(final_path) or "."
    with tempfile.NamedTemporaryFile(delete=False, dir=d, prefix=".tmp_", suffix=".npz") as f:
        if compressed:
            np.savez_compressed(f, **arrays)
        else:
            np.savez(f, **arrays)
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())
        tmp = f.name
    os.replace(tmp, final_path)


# ---------- 几何收集与去重 ----------

def _collect_static_geometry(comm, msh, tol_rel=1e-14):
    """收集网格几何信息到rank 0"""
    imV = msh.topology.index_map(0)
    nlocV = imV.size_local
    nallV = nlocV + imV.num_ghosts
    Xloc_all = msh.geometry.x[:nallV, :2].copy()

    tdim = msh.topology.dim
    imC = msh.topology.index_map(tdim)
    nlocC = imC.size_local
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    arr, offs = c2v.array, c2v.offsets
    cells_vloc = [arr[offs[i]:offs[i+1]].copy() for i in range(nlocC)]

    pack = (Xloc_all, nlocV, cells_vloc, nlocC)
    packs = comm.gather(pack, root=0)
    if comm.rank != 0:
        return None

    key_to_gid = {}
    Xg = []
    l2g_list = []
    tri_all = []
    owner_all = []
    c_off = 0
    for (Xr, nlocVr, cells_r, nlocCr) in packs:
        span = (Xr.max(axis=0) - Xr.min(axis=0)).max()
        tol = max(tol_rel * max(span, 1.0), 5e-16)
        qr = np.round(Xr / tol).astype(np.int64)
        keys = [tuple(xy) for xy in qr]

        l2g = np.empty(len(keys), dtype=np.int64)
        for i, k in enumerate(keys):
            gid = key_to_gid.get(k, -1)
            if gid < 0:
                gid = len(Xg)
                key_to_gid[k] = gid
                Xg.append(Xr[i])
            l2g[i] = gid
        l2g_list.append(l2g)

        for cid_local, vv in enumerate(cells_r):
            gl = l2g[vv]
            if gl.size < 3:
                continue
            for k in range(1, gl.size - 1):
                tri_all.append([int(gl[0]), int(gl[k]), int(gl[k+1])])
                owner_all.append(int(c_off + cid_local))
        c_off += nlocCr

    Xg = np.asarray(Xg, dtype=float)
    TRI = np.asarray(tri_all, dtype=np.int32)
    OWNER = np.asarray(owner_all, dtype=np.int64)

    TRI, OWNER = _sanitize_tris(Xg, TRI, OWNER)
    return Xg, TRI, OWNER, l2g_list


def _sanitize_tris(X, TRI, OWNER, area_tol=1e-30):
    """去除退化和重复的三角形"""
    TRI = TRI.astype(np.int64, copy=False)
    keep_distinct = (TRI[:,0] != TRI[:,1]) & (TRI[:,0] != TRI[:,2]) & (TRI[:,1] != TRI[:,2])
    TRI = TRI[keep_distinct]; OWNER = OWNER[keep_distinct]
    P = X[TRI]
    areas = 0.5 * np.abs(
        (P[:,1,0] - P[:,0,0]) * (P[:,2,1] - P[:,0,1])
      - (P[:,2,0] - P[:,0,0]) * (P[:,1,1] - P[:,0,1])
    )
    keep_area = areas > area_tol
    TRI = TRI[keep_area]; OWNER = OWNER[keep_area]
    keys = np.sort(TRI, axis=1)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    TRI = TRI[uniq_idx]; OWNER = OWNER[uniq_idx]
    return TRI.astype(np.int32, copy=False), OWNER


# ---------- 数据收集辅助函数 ----------

def _gather_nodal_owned_to_root(comm, V_scalar, func, l2g_list):
    """收集节点数据到rank 0"""
    imV = V_scalar.mesh.topology.index_map(0)
    nlocV = imV.size_local
    verts_loc = np.arange(nlocV, dtype=np.int32)
    dofs = fem.locate_dofs_topological(V_scalar, 0, verts_loc)
    vals = func.x.array[dofs].copy()
    packs = comm.gather(vals, root=0)
    if comm.rank != 0:
        return None
    Nv = max(l.max() for l in l2g_list) + 1
    out = np.empty(Nv, dtype=float)
    for r, vals_r in enumerate(packs):
        l2g = l2g_list[r]
        out[l2g[:len(vals_r)]] = vals_r
    return out


def _gather_cell_DG0_owned_to_root(comm, V_ten, func):
    """收集DG0单元数据到rank 0"""
    dm = func.function_space.dofmap
    imC = dm.index_map
    nlocC = imC.size_local

    m = 0
    if nlocC > 0:
        m = len(dm.cell_dofs(0))
    if m <= 0:
        try:
            m = int(func.function_space.element.value_size)
        except Exception:
            m = 0
    if m <= 0:
        arr = func.x.array
        rows_guess = max(nlocC + int(getattr(imC, "num_ghosts", 0)), 1)
        m = max(arr.size // rows_guess, 1)

    arr = func.x.array
    rows_total = arr.size // m
    A_owned = arr.reshape(rows_total, m)[:nlocC, :].copy()
    packs = comm.gather(A_owned, root=0)
    if comm.rank != 0:
        return None
    return np.vstack(packs)


def _extract_xx_yy_xy(arr4):
    """从4分量数组提取xx, yy, xy"""
    xx = arr4[:, 0]
    xy = 0.5 * (arr4[:, 1] + arr4[:, 2])
    yy = arr4[:, 3]
    return xx, yy, xy


# ---------- 规则网格工具 ----------

def _global_bbox_2d(msh, margin=1e-10):
    """计算全局边界框"""
    x_local = msh.geometry.x[:, :2]
    xmin_l = x_local.min(axis=0)
    xmax_l = x_local.max(axis=0)
    xmin = np.array([msh.comm.allreduce(float(xmin_l[0]), op=MPI.MIN),
                     msh.comm.allreduce(float(xmin_l[1]), op=MPI.MIN)])
    xmax = np.array([msh.comm.allreduce(float(xmax_l[0]), op=MPI.MAX),
                     msh.comm.allreduce(float(xmax_l[1]), op=MPI.MAX)])
    return xmin, xmax


def _build_regular_grid(comm, msh, Nx, Ny):
    """构建规则网格"""
    xmin, xmax = _global_bbox_2d(msh)
    if comm.rank == 0:
        gx = np.linspace(xmin[0], xmax[0], Nx)
        gy = np.linspace(xmin[1], xmax[1], Ny)
        X, Y = np.meshgrid(gx, gy, indexing="xy")
        P = np.column_stack([X.ravel(order="C"), Y.ravel(order="C"), np.zeros(X.size)])
    else:
        gx = gy = X = Y = P = None
    gx = comm.bcast(gx, root=0)
    gy = comm.bcast(gy, root=0)
    P = comm.bcast(P, root=0)
    return gx, gy, X, Y, P


def _value_size(func_or_space):
    """获取函数/空间的值大小"""
    if hasattr(func_or_space, "function_space"):
        fs = func_or_space.function_space
    else:
        fs = func_or_space
    try:
        vs = int(getattr(fs.element, "value_size", 0))
        if vs > 0:
            return vs
    except Exception:
        pass
    try:
        ufl_el = fs.ufl_element()
        vshape = getattr(ufl_el, "value_shape", None)
        return int(np.prod(vshape)) if vshape else 1
    except Exception:
        return 1


def _sample_function_on_points(msh, func, P_all):
    """在给定点上采样函数（适用于CG1标量/向量）"""
    tdim = msh.topology.dim
    tree = geometry.bb_tree(msh, tdim)
    hits = geometry.compute_collisions_points(tree, P_all)
    cells = geometry.compute_colliding_cells(msh, hits, P_all)

    m = _value_size(func)
    out = np.full((P_all.shape[0], m), np.nan, dtype=float)
    for i in range(P_all.shape[0]):
        cl = cells.links(i)
        if len(cl) == 0:
            continue
        cell = int(cl[0])
        try:
            val = func.eval(P_all[i], cell)
            vv = np.asarray(val, dtype=float).reshape(-1)
            if vv.size == m:
                out[i, :] = vv
            elif vv.size == 1 and m > 1:
                out[i, 0] = float(vv[0])
        except Exception:
            pass
    return out


def _gather_grid_values(comm, arr_local):
    """收集网格值到rank 0"""
    packs = comm.gather(arr_local, root=0)
    if comm.rank != 0:
        return None
    stacked = np.stack(packs, axis=0)
    _, N, m = stacked.shape
    out = np.full((N, m), np.nan, dtype=float)
    for j in range(m):
        col = stacked[:, :, j]
        mask = np.isfinite(col)
        has = mask.any(axis=0)
        first_idx = np.argmax(mask, axis=0)
        idx = np.arange(N)
        out[has, j] = col[first_idx[has], idx[has]]
    return out


def _build_DG0_to_grid_interpolator(comm, msh, V0, P_all, grid_shape):
    """
    预计算从DG0单元到规则网格点的映射
    """
    if comm.rank == 0:
        tdim = msh.topology.dim
        tree = geometry.bb_tree(msh, tdim)
        hits = geometry.compute_collisions_points(tree, P_all)
        cells_per_point = geometry.compute_colliding_cells(msh, hits, P_all)
        
        point_to_cell = np.full(P_all.shape[0], -1, dtype=np.int32)
        for i in range(P_all.shape[0]):
            cl = cells_per_point.links(i)
            if len(cl) > 0:
                point_to_cell[i] = int(cl[0])
        
        return {
            'point_to_cell': point_to_cell,
            'grid_shape': grid_shape,
            'n_points': P_all.shape[0]
        }
    return None


def _fast_sample_DG0_to_grid(comm, func_DG0, interp_data):
    """
    使用预计算的映射快速采样DG0函数到规则网格
    """
    arr_local = func_DG0.x.array.copy()
    imC = func_DG0.function_space.dofmap.index_map
    nlocC = imC.size_local
    arr_owned = arr_local[:nlocC]
    
    packs = comm.gather(arr_owned, root=0)
    
    if comm.rank != 0:
        return None
    
    arr_global = np.concatenate(packs)
    
    point_to_cell = interp_data['point_to_cell']
    grid_values = np.full(interp_data['n_points'], np.nan, dtype=float)
    
    valid_mask = point_to_cell >= 0
    grid_values[valid_mask] = arr_global[point_to_cell[valid_mask]]
    
    Ny, Nx = interp_data['grid_shape']
    return grid_values.reshape(Ny, Nx)


# ---------- 可视化工具 ----------

def write_png_d_u(out_dir, prefix, step_idx, grid_x, grid_y, d_2d, u1_2d, u2_2d, dpi=200, cmap="viridis"):
    """
    保存 1x3 的图片：d, u_x, u_y
    
    Parameters:
    -----------
    out_dir : str
        输出目录
    prefix : str
        文件名前缀
    step_idx : int
        步骤索引
    grid_x, grid_y : array
        网格坐标
    d_2d, u1_2d, u2_2d : array (Ny, Nx)
        相场和位移分量
    dpi : int
        图像分辨率
    cmap : str
        颜色映射
    
    Returns:
    --------
    fn : str
        生成的文件名
    """
    fn = os.path.join(out_dir, f"{prefix}_d_u_{step_idx:04d}.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    
    # 子图1: 相场 d
    im0 = axes[0].imshow(
        d_2d, origin="lower",
        extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
        cmap=cmap, interpolation="bilinear"
    )
    axes[0].set_title(f"Phase field d (step {step_idx})", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 子图2: x方向位移 u1
    im1 = axes[1].imshow(
        u1_2d, origin="lower",
        extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
        cmap="RdBu_r", interpolation="bilinear"
    )
    axes[1].set_title(f"Displacement u_x (step {step_idx})", fontsize=12)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 子图3: y方向位移 u2
    im2 = axes[2].imshow(
        u2_2d, origin="lower",
        extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
        cmap="RdBu_r", interpolation="bilinear"
    )
    axes[2].set_title(f"Displacement u_y (step {step_idx})", fontsize=12)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig.savefig(fn, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fn
