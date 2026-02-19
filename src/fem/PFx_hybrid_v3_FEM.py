"""
FEM求解模块 - 相场断裂力学求解

提供FEM求解功能，给定相场d、边界条件和材料参数，求解应力场
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
import dolfinx.fem.petsc as fem_petsc
from dolfinx.nls.petsc import NewtonSolver
import ufl

# ---------- UFL utilities ----------
try:
    ufl_abs = ufl.abs
except AttributeError:
    from ufl.algebra import Abs as _UFLAbs
    def ufl_abs(x): return _UFLAbs(x)

def mac_pos(x):  # <x>_+ = (x + |x|)/2
    return 0.5 * (x + ufl_abs(x))

def eps(u):  # symmetric gradient
    return ufl.sym(ufl.grad(u))

def spectral_eps_pos(e):
    """谱分解应变张量的正部分"""
    I = ufl.Identity(2)
    a = e[0, 0]; b = e[0, 1]; c = e[1, 1]
    tr = a + c
    delta = ufl.sqrt(((a - c) * 0.5) ** 2 + b ** 2 + 1e-12)
    l1 = 0.5 * tr + delta
    l2 = 0.5 * tr - delta
    denom = l1 - l2
    P1 = ufl.conditional(ufl.gt(ufl_abs(denom), 1e-12), (e - l2 * I) / denom, 0.5 * I)
    P2 = I - P1
    l1p = mac_pos(l1)
    l2p = mac_pos(l2)
    e_pos = l1p * P1 + l2p * P2
    return e_pos, l1p, l2p, tr

def sigma_plus(u, lam, mu):
    """应力张量的正部分（拉伸）"""
    e = eps(u)
    e_pos, _, _, tr = spectral_eps_pos(e)
    return 2*mu*e_pos + lam*mac_pos(tr)*ufl.Identity(2)

def sigma_minus(u, lam, mu):
    """应力张量的负部分（压缩）"""
    e = eps(u)
    e_pos, _, _, tr = spectral_eps_pos(e)
    e_neg = e - e_pos
    return 2*mu*e_neg + lam*(ufl.tr(e) - mac_pos(tr))*ufl.Identity(2)

def psi_plus(u, lam, mu):
    """应变能密度的正部分"""
    e = eps(u)
    e_pos, _, _, tr = spectral_eps_pos(e)
    return 0.5 * lam * (mac_pos(tr) ** 2) + mu * ufl.inner(e_pos, e_pos)

def l2_project(V, expr):
    """使用L2投影计算函数"""
    v = ufl.TestFunction(V)
    w = fem.Function(V)
    a = fem.form(ufl.inner(ufl.TrialFunction(V), v) * ufl.dx)
    L = fem.form(ufl.inner(expr, v) * ufl.dx)
    A = fem_petsc.assemble_matrix(a); A.assemble()
    b = fem_petsc.assemble_vector(L)
    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    ksp.setFromOptions()
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    ksp.destroy(); A.destroy(); b.destroy()
    return w


# ---------- 函数空间和边界条件 ----------

def create_function_spaces(msh):
    """
    创建所有需要的函数空间
    
    Returns:
    --------
    dict : 包含所有函数空间的字典
    """
    # 创建必要的拓扑连接
    tdim = msh.topology.dim
    msh.topology.create_connectivity(0, tdim)
    msh.topology.create_connectivity(tdim, 0)
    
    return {
        'V_u': fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,))),
        'V_d': fem.functionspace(msh, ("Lagrange", 1)),
        'V_ten': fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, msh.geometry.dim))),
        'V0': fem.functionspace(msh, ("DG", 0))
    }


def locate_top_bottom_facets(msh):
    """
    定位上下边界面
    
    Returns:
    --------
    top_facets, bottom_facets : 上下边界面的索引
    """
    y = msh.geometry.x[:, 1]
    ymax = msh.comm.allreduce(float(y.max()), op=MPI.MAX)
    ymin = msh.comm.allreduce(float(y.min()), op=MPI.MIN)
    fdim = msh.topology.dim - 1
    is_bottom = lambda x: np.isclose(x[1], ymin)
    is_top = lambda x: np.isclose(x[1], ymax)
    bottom = mesh.locate_entities_boundary(msh, fdim, is_bottom)
    top = mesh.locate_entities_boundary(msh, fdim, is_top)
    return top, bottom


def setup_boundary_conditions(V_u, msh, top_disp_value=0.0):
    """
    设置边界条件
    
    Parameters:
    -----------
    V_u : dolfinx.fem.FunctionSpace
        位移函数空间
    msh : dolfinx.mesh.Mesh
        网格
    top_disp_value : float
        顶部位移值
    
    Returns:
    --------
    bcs_u : list
        边界条件列表
    top_y_fun : dolfinx.fem.Function
        顶部位移函数（用于动态调整）
    """
    top_facets, bottom_facets = locate_top_bottom_facets(msh)
    fdim = V_u.mesh.topology.dim - 1
    
    # 底部y方向固定
    dofs_by = fem.locate_dofs_topological(V_u.sub(1), fdim, bottom_facets)
    bc_by = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_by, V_u.sub(1))
    
    # 左侧x方向固定
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_lx = fem.locate_dofs_topological(V_u.sub(0), fdim, left_facets)
    bc_lx = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_lx, V_u.sub(0))
    
    # 顶部y方向位移控制
    W1, _ = V_u.sub(1).collapse()
    top_y_fun = fem.Function(W1)
    dofs_ty = fem.locate_dofs_topological((V_u.sub(1), W1), fdim, top_facets)
    bc_ty = fem.dirichletbc(top_y_fun, dofs_ty, V_u.sub(1))
    top_y_fun.x.array[:] = top_disp_value
    
    return [bc_by, bc_lx, bc_ty], top_y_fun


class FEMSolver:
    """
    FEM求解器类 - 封装力学求解的所有状态和方法
    """
    def __init__(self, msh, E, nu, k_reg, top_disp_value=0.0):
        """
        初始化FEM求解器
        
        Parameters:
        -----------
        msh : dolfinx.mesh.Mesh
            网格
        E : float
            杨氏模量
        nu : float
            泊松比
        k_reg : float
            正则化参数
        top_disp_value : float
            顶部位移初始值
        """
        self.msh = msh
        self.comm = msh.comm
        self.E = E
        self.nu = nu
        self.k_reg = k_reg
        
        # 计算Lamé常数
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        
        # 创建函数空间
        spaces = create_function_spaces(msh)
        self.V_u = spaces['V_u']
        self.V_d = spaces['V_d']
        self.V0 = spaces['V0']
        
        # 设置边界条件
        self.bcs_u, self.top_y_fun = setup_boundary_conditions(
            self.V_u, msh, top_disp_value
        )
        
        # 创建位移函数
        self.u = fem.Function(self.V_u, name="u")
        
        # 创建相场占位符
        self.d_placeholder = fem.Function(self.V_d, name="d")
        
        # 设置求解器
        self._setup_solver()
    
    def _setup_solver(self):
        """设置非线性求解器"""
        # 定义变分形式
        du = ufl.TrialFunction(self.V_u)
        v_u = ufl.TestFunction(self.V_u)
        
        # 退化函数
        gdeg = lambda d_: (1 - d_)**2 + self.k_reg
        
        # 应力
        stress_u = gdeg(self.d_placeholder) * sigma_plus(self.u, self.lam, self.mu) + \
                   sigma_minus(self.u, self.lam, self.mu)
        
        # 变分形式
        F_u = ufl.inner(stress_u, eps(v_u)) * ufl.dx
        J_u = ufl.derivative(F_u, self.u, du)
        
        # 创建非线性问题
        self.problem_u = fem_petsc.NonlinearProblem(F_u, self.u, bcs=self.bcs_u, J=J_u)
        self.solver_u = NewtonSolver(self.comm, self.problem_u)
        
        # 设置求解器参数
        self.solver_u.convergence_criterion = "residual"
        self.solver_u.rtol = 1e-8
        self.solver_u.atol = 1e-9
        self.solver_u.max_it = 300
        
        ksp = self.solver_u.krylov_solver
        try:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
        except Exception:
            ksp.setType("gmres")
            ksp.getPC().setType("gamg")
        ksp.setFromOptions()
    
    def solve(self, d_array):
        """
        求解力学方程
        
        Parameters:
        -----------
        d_array : numpy.ndarray
            相场值数组（在位移函数空间的自由度上）
        
        Returns:
        --------
        success : bool
            求解是否成功
        u : dolfinx.fem.Function
            位移场
        """
        # 更新相场值
        self.d_placeholder.x.array[:] = d_array
        self.d_placeholder.x.scatter_forward()
        
        # 求解
        try:
            _, ok_local = self.solver_u.solve(self.u)
            self.u.x.scatter_forward()
        except RuntimeError:
            ok_local = False
        
        ok = bool(self.comm.allreduce(int(bool(ok_local)), op=MPI.LAND))
        
        return ok, self.u
    
    def compute_stress_strain(self, u=None, d_array=None):
        """
        计算应力和应变场
        
        Parameters:
        -----------
        u : dolfinx.fem.Function, optional
            位移场（若为None则使用最后求解的位移）
        d_array : numpy.ndarray, optional
            相场值（若为None则使用当前的d值）
        
        Returns:
        --------
        dict : 包含应力和应变分量的字典
            'exx', 'eyy', 'exy': 应变分量 (DG0)
            'sxx', 'syy', 'sxy': 应力分量 (DG0)
        """
        if u is None:
            u = self.u
        
        if d_array is not None:
            self.d_placeholder.x.array[:] = d_array
            self.d_placeholder.x.scatter_forward()
        
        # 计算应变
        E_expr = eps(u)
        exx0 = l2_project(self.V0, E_expr[0, 0])
        eyy0 = l2_project(self.V0, E_expr[1, 1])
        exy0 = l2_project(self.V0, 0.5 * (E_expr[0, 1] + E_expr[1, 0]))
        
        # 计算应力（包含退化）
        gdeg = (1 - self.d_placeholder)**2 + self.k_reg
        sigma_eff = gdeg * sigma_plus(u, self.lam, self.mu) + sigma_minus(u, self.lam, self.mu)
        
        sxx0 = l2_project(self.V0, sigma_eff[0, 0])
        syy0 = l2_project(self.V0, sigma_eff[1, 1])
        sxy0 = l2_project(self.V0, 0.5 * (sigma_eff[0, 1] + sigma_eff[1, 0]))
        
        return {
            'exx': exx0,
            'eyy': eyy0,
            'exy': exy0,
            'sxx': sxx0,
            'syy': syy0,
            'sxy': sxy0
        }
    
    def update_history_field(self, H, u=None):
        """
        更新历史场（最大应变能密度）
        
        Parameters:
        -----------
        H : dolfinx.fem.Function
            历史场（会被原地更新）
        u : dolfinx.fem.Function, optional
            位移场（若为None则使用最后求解的位移）
        
        Returns:
        --------
        H : dolfinx.fem.Function
            更新后的历史场
        """
        if u is None:
            u = self.u
        
        V_d = H.function_space
        Hp = l2_project(V_d, psi_plus(u, self.lam, self.mu))
        H.x.array[:] = np.maximum(H.x.array, Hp.x.array)
        H.x.scatter_forward()
        
        return H


# ---------- 完全封装的网格求解器 ----------

class FEMGridSolver:
    """
    完全封装的FEM网格求解器
    
    主函数只需操作numpy数组，所有FEM和插值操作都在内部完成
    """
    def __init__(self, msh, grid_nx, grid_ny, E=210.0, nu=0.3, k_reg=1e-5, 
                 Gc=2.7e-3, top_disp_value=0.0):
        """
        初始化FEM网格求解器
        
        Parameters:
        -----------
        msh : dolfinx.mesh.Mesh
            网格
        grid_nx, grid_ny : int
            规则网格尺寸
        E, nu, k_reg : float
            材料参数
        Gc : float
            断裂能
        top_disp_value : float
            顶部位移初始值
        """
        from PFx_hybrid_v3_utils import _build_regular_grid, _build_DG0_to_grid_interpolator
        from scipy.interpolate import RegularGridInterpolator
        
        self.msh = msh
        self.comm = msh.comm
        self.rank = msh.comm.rank
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.Gc = Gc
        
        # 创建函数空间
        spaces = create_function_spaces(msh)
        self.V_u = spaces['V_u']
        self.V_d = spaces['V_d']
        self.V0 = spaces['V0']
        
        # 创建FEM求解器
        self.fem_solver = FEMSolver(msh, E, nu, k_reg, top_disp_value)
        
        # 创建规则网格
        self.grid_x, self.grid_y, _, _, P_all = _build_regular_grid(
            self.comm, msh, grid_nx, grid_ny
        )
        self.P_all = P_all
        
        # 预计算DG0到网格的映射
        self.DG0_interp_data = _build_DG0_to_grid_interpolator(
            self.comm, msh, self.V0, P_all, (grid_ny, grid_nx)
        )
        
        # 设置标量场插值器
        if self.rank != 0:
            self.grid_x = self.comm.bcast(None, root=0)
            self.grid_y = self.comm.bcast(None, root=0)
        else:
            self.comm.bcast(self.grid_x, root=0)
            self.comm.bcast(self.grid_y, root=0)
        
        imV = self.V_d.mesh.topology.index_map(0)
        nlocV = imV.size_local
        verts_loc = np.arange(nlocV, dtype=np.int32)
        self.dofs_loc = fem.locate_dofs_topological(self.V_d, 0, verts_loc)
        x_local_raw = self.V_d.mesh.geometry.x[:nlocV, :2]
        self.x_local = np.column_stack([x_local_raw[:, 1], x_local_raw[:, 0]])
        
        dummy_grid = np.ones((grid_ny, grid_nx))
        self.interp_obj = RegularGridInterpolator(
            (self.grid_y, self.grid_x), dummy_grid,
            method='linear', bounds_error=False, fill_value=0.0
        )
        
        # 创建内部使用的fem.Function
        self.d_func = fem.Function(self.V_d, name="d")
        self.H_func = fem.Function(self.V_d, name="H")
        self.d_pre_func = fem.Function(self.V_d, name="d_pre")
        
        if self.rank == 0:
            print(f"[FEMGridSolver] Initialized with grid {grid_ny}x{grid_nx}")
    
    def set_top_displacement(self, disp_value):
        """设置顶部位移"""
        self.fem_solver.top_y_fun.x.array[:] = disp_value
    
    def solve_and_sample(self, d_grid, d_pre_grid, H_grid):
        """
        求解FEM并采样到规则网格（完全封装，只操作numpy数组）
        
        Parameters:
        -----------
        d_grid : np.ndarray (Ny, Nx) or None
            相场网格值（rank 0），其他rank传None
        d_pre_grid : np.ndarray (Ny, Nx) or None
            前一步相场网格值（rank 0），其他rank传None
        H_grid : np.ndarray (Ny, Nx) or None
            历史场网格值（rank 0），其他rank传None
        
        Returns:
        --------
        success : bool
            求解是否成功
        grid_fields : dict or None
            规则网格上的场（仅rank==0返回），包含：
            'exx', 'eyy', 'exy', 'sxx', 'syy', 'sxy', 'H', 'd_pre' (所有为 Ny×Nx)
        """
        from PFx_hybrid_v3_utils import _sample_function_on_points, _gather_grid_values, _fast_sample_DG0_to_grid
        
        # 1. 将网格数据插值到FEM节点
        if self.rank == 0:
            d_grid_bcast = np.nan_to_num(d_grid, nan=0.0)
        else:
            d_grid_bcast = np.empty((self.grid_ny, self.grid_nx), dtype=np.float32)
        
        self.comm.Bcast(d_grid_bcast, root=0)
        self.interp_obj.values = d_grid_bcast
        d_interp_local = self.interp_obj(self.x_local)
        self.d_func.x.array[self.dofs_loc] = d_interp_local
        self.d_func.x.scatter_forward()
        
        # 2. 求解FEM
        success, u = self.fem_solver.solve(self.d_func.x.array)
        if not success:
            return False, None
        
        # 3. 计算应力应变
        ss_fields = self.fem_solver.compute_stress_strain(u, self.d_func.x.array)
        
        # 4. 采样到规则网格
        if self.rank == 0:
            exx_grid = _fast_sample_DG0_to_grid(self.comm, ss_fields['exx'], self.DG0_interp_data)
            eyy_grid = _fast_sample_DG0_to_grid(self.comm, ss_fields['eyy'], self.DG0_interp_data)
            exy_grid = _fast_sample_DG0_to_grid(self.comm, ss_fields['exy'], self.DG0_interp_data)
            sxx_grid = _fast_sample_DG0_to_grid(self.comm, ss_fields['sxx'], self.DG0_interp_data)
            syy_grid = _fast_sample_DG0_to_grid(self.comm, ss_fields['syy'], self.DG0_interp_data)
            sxy_grid = _fast_sample_DG0_to_grid(self.comm, ss_fields['sxy'], self.DG0_interp_data)
        else:
            _fast_sample_DG0_to_grid(self.comm, ss_fields['exx'], None)
            _fast_sample_DG0_to_grid(self.comm, ss_fields['eyy'], None)
            _fast_sample_DG0_to_grid(self.comm, ss_fields['exy'], None)
            _fast_sample_DG0_to_grid(self.comm, ss_fields['sxx'], None)
            _fast_sample_DG0_to_grid(self.comm, ss_fields['syy'], None)
            _fast_sample_DG0_to_grid(self.comm, ss_fields['sxy'], None)
            return True, None
        
        # 5. 采样H和d_pre（从网格值）
        grid_fields = {
            'exx': exx_grid,
            'eyy': eyy_grid,
            'exy': exy_grid,
            'sxx': sxx_grid,
            'syy': syy_grid,
            'sxy': sxy_grid,
            'H': H_grid,
            'd_pre': d_pre_grid
        }
        
        return True, grid_fields
    
    def update_history(self, H_grid):
        """
        更新历史场
        
        Parameters:
        -----------
        H_grid : np.ndarray (Ny, Nx) or None
            当前历史场网格值（rank 0），其他rank传None
        
        Returns:
        --------
        H_grid_new : np.ndarray (Ny, Nx) or None
            更新后的历史场（仅rank 0返回）
        """
        from PFx_hybrid_v3_utils import _sample_function_on_points, _gather_grid_values
        
        # 插值H到FEM节点
        if self.rank == 0:
            H_grid_bcast = np.nan_to_num(H_grid, nan=0.0)
        else:
            H_grid_bcast = np.empty((self.grid_ny, self.grid_nx), dtype=np.float32)
        
        self.comm.Bcast(H_grid_bcast, root=0)
        self.interp_obj.values = H_grid_bcast
        H_interp_local = self.interp_obj(self.x_local)
        self.H_func.x.array[self.dofs_loc] = H_interp_local
        self.H_func.x.scatter_forward()
        
        # 更新历史场
        self.fem_solver.update_history_field(self.H_func, self.fem_solver.u)
        
        # 采样回网格
        H_local = _sample_function_on_points(self.msh, self.H_func, self.P_all)
        H_all = _gather_grid_values(self.comm, H_local)
        
        if self.rank == 0:
            return H_all[:, 0].reshape(self.grid_ny, self.grid_nx)
        return None


# ---------- 高级接口：一站式FEM求解和网格采样 ----------

def solve_and_sample_on_grid(fem_solver, d_array, d_pre_array, H_array, 
                              grid_interp_data, rank=0):
    """
    高级接口：求解FEM并采样到规则网格（封装所有复杂操作）
    
    Parameters:
    -----------
    fem_solver : FEMSolver
        FEM求解器实例
    d_array : numpy.ndarray
        相场值数组（在节点上）
    d_pre_array : numpy.ndarray
        前一步的相场值（在节点上）
    H_array : numpy.ndarray
        历史场值（在节点上）
    grid_interp_data : dict
        网格插值所需的预计算数据，包含：
        - 'DG0_interp_data': DG0到网格的映射
        - 'P_all': 规则网格点坐标
        - 'msh': 网格
        - 'V_d': 相场函数空间
        - 'scalar_interp_data': 标量场插值器
    rank : int
        MPI rank
    
    Returns:
    --------
    success : bool
        求解是否成功
    grid_fields : dict or None
        规则网格上的场，仅在rank==0时返回，包含：
        - 'exx', 'eyy', 'exy': 应变分量 (Ny, Nx)
        - 'sxx', 'syy', 'sxy': 应力分量 (Ny, Nx)
        - 'H': 历史场 (Ny, Nx)
        - 'd_pre': 前一步相场 (Ny, Nx)
    """
    from PFx_hybrid_v3_utils import (
        _sample_function_on_points, _gather_grid_values, _fast_sample_DG0_to_grid
    )
    
    # 1. 求解力学方程
    success, u = fem_solver.solve(d_array)
    if not success:
        return False, None
    
    # 2. 计算应力应变场
    ss_fields = fem_solver.compute_stress_strain(u, d_array)
    
    # 3. 采样到规则网格
    comm = fem_solver.comm
    DG0_interp_data = grid_interp_data['DG0_interp_data']
    P_all = grid_interp_data['P_all']
    msh = grid_interp_data['msh']
    V_d = grid_interp_data['V_d']
    
    # 采样DG0场（应力应变）
    if rank == 0:
        exx_grid = _fast_sample_DG0_to_grid(comm, ss_fields['exx'], DG0_interp_data)
        eyy_grid = _fast_sample_DG0_to_grid(comm, ss_fields['eyy'], DG0_interp_data)
        exy_grid = _fast_sample_DG0_to_grid(comm, ss_fields['exy'], DG0_interp_data)
        sxx_grid = _fast_sample_DG0_to_grid(comm, ss_fields['sxx'], DG0_interp_data)
        syy_grid = _fast_sample_DG0_to_grid(comm, ss_fields['syy'], DG0_interp_data)
        sxy_grid = _fast_sample_DG0_to_grid(comm, ss_fields['sxy'], DG0_interp_data)
    else:
        _fast_sample_DG0_to_grid(comm, ss_fields['exx'], None)
        _fast_sample_DG0_to_grid(comm, ss_fields['eyy'], None)
        _fast_sample_DG0_to_grid(comm, ss_fields['exy'], None)
        _fast_sample_DG0_to_grid(comm, ss_fields['sxx'], None)
        _fast_sample_DG0_to_grid(comm, ss_fields['syy'], None)
        _fast_sample_DG0_to_grid(comm, ss_fields['sxy'], None)
        return True, None
    
    # 采样标量场（H, d_pre）
    H_func = fem.Function(V_d)
    H_func.x.array[:] = H_array
    d_pre_func = fem.Function(V_d)
    d_pre_func.x.array[:] = d_pre_array
    
    H_local = _sample_function_on_points(msh, H_func, P_all)
    dpre_local = _sample_function_on_points(msh, d_pre_func, P_all)
    H_all = _gather_grid_values(comm, H_local)
    dpre_all = _gather_grid_values(comm, dpre_local)
    
    # 整理结果
    Ny, Nx = DG0_interp_data['grid_shape']
    grid_fields = {
        'exx': exx_grid,
        'eyy': eyy_grid,
        'exy': exy_grid,
        'sxx': sxx_grid,
        'syy': syy_grid,
        'sxy': sxy_grid,
        'H': H_all[:, 0].reshape(Ny, Nx),
        'd_pre': dpre_all[:, 0].reshape(Ny, Nx)
    }
    
    return True, grid_fields
    