"""
CEC 2019 Multi-objective Multi-task Optimization Benchmark (CEC19-MaTMO)
Python port chính xác từ MATLAB gốc (MTO-Platform - intLyc)

Cấu trúc bài toán:
  - Mỗi MaTMO có 50 tasks (mặc định), 2 objectives
  - x[0]       : biến position trên Pareto front (∈ [0,1])
  - x[1:]      : biến decision (bị shift + rotate bởi g-function)
  - Output     : [f1, f2]

Pareto front shapes:
  MaTMO1: circle   | MaTMO2: concave | MaTMO3: concave
  MaTMO4: circle   | MaTMO5: convex  | MaTMO6: circle

Cách dùng:
    from cec19_matmo import CEC19_MaTMO

    problem = CEC19_MaTMO(problem_id=1, data_dir='.')
    tasks = problem.tasks          # list of 50 Task objects

    x = np.random.rand(tasks[0].dim)        # x ∈ [lb, ub] (mixed bounds)
    f = tasks[0].evaluate(x)                # → np.array([f1, f2])

    # Hoặc evaluate nhiều cá thể cùng lúc (N x dim)
    X = np.random.rand(100, tasks[0].dim)
    F = tasks[0].evaluate_batch(X)          # → (100, 2)
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable


# ──────────────────────────────────────────────────────────────
# Cấu hình 6 MaTOPs (đúng theo benchmark_CEC19_MaTMO.m)
# ──────────────────────────────────────────────────────────────
PROBLEM_CONFIGS = {
    # id: (dim, lb_main, ub_main, gType,       f1Type,   hType)
    1: (50, -100, 100, 'Sphere',     'linear', 'circle'),
    2: (50, -100, 100, 'Mean',       'linear', 'concave'),
    3: (10,   -5,   5, 'Rosenbrock', 'linear', 'concave'),
    4: (50,   -2,   2, 'Rastrigin',  'linear', 'circle'),
    5: (50,   -1,   1, 'Ackley',     'linear', 'convex'),
    6: (50,  -50,  50, 'Griewank',   'linear', 'circle'),
}


# ──────────────────────────────────────────────────────────────
# G-functions (port từ evalGfunction trong getFun_CEC19_MaTMO.m)
# x: (N, d) numpy array
# ──────────────────────────────────────────────────────────────

def g_sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2, axis=1)


def g_mean(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return 9 * np.sum(np.abs(x), axis=1) / d


def g_rosenbrock(x: np.ndarray) -> np.ndarray:
    t = np.zeros(x.shape[0])
    for i in range(x.shape[1] - 1):
        t += 100 * (x[:, i] ** 2 - x[:, i + 1]) ** 2 + (1 - x[:, i]) ** 2
    return t


def g_ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    sum1 = np.sum(x ** 2, axis=1) / d
    sum2 = np.sum(np.cos(2 * np.pi * x), axis=1) / d
    return -20 * np.exp(-0.2 * np.sqrt(sum1)) - np.exp(sum2) + 20 + np.e


def g_griewank(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    t = np.sqrt(np.arange(1, d + 1))
    sum1 = np.sum(x ** 2, axis=1)
    prod1 = np.prod(np.cos(x / t), axis=1)
    return 1 + sum1 / 4000 - prod1


def g_rastrigin(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    a = 10 * d
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1) + a


G_FUNCTIONS = {
    'Sphere':     g_sphere,
    'Mean':       g_mean,
    'Rosenbrock': g_rosenbrock,
    'Ackley':     g_ackley,
    'Griewank':   g_griewank,
    'Rastrigin':  g_rastrigin,
}


# ──────────────────────────────────────────────────────────────
# evalF1 — tính f1 từ x[:,0] (biến position)
# ──────────────────────────────────────────────────────────────

def eval_f1(x_pos: np.ndarray, f1_type: str) -> np.ndarray:
    """x_pos: (N, 1) — chỉ dùng cột đầu tiên"""
    x1 = x_pos[:, 0]  # shape (N,)
    if f1_type == 'linear':
        return x1  # x[:,0] ∈ [0,1] → f1 ∈ [0,1]
    else:
        r = np.sqrt(np.sum(x_pos ** 2, axis=1))
        return 1 - np.exp(-4 * r) * np.sin(5 * np.pi * r) ** 4


# ──────────────────────────────────────────────────────────────
# evalH — hàm kết hợp f1 + g → f2
# ──────────────────────────────────────────────────────────────

def eval_h(f1: np.ndarray, g: np.ndarray, h_type: str) -> np.ndarray:
    if h_type == 'convex':
        return 1 - np.sqrt(f1 / g)
    else:  # concave (cả circle dùng công thức này qua f1/g)
        return 1 - (f1 / g) ** 2


# ──────────────────────────────────────────────────────────────
# Task class
# ──────────────────────────────────────────────────────────────

@dataclass
class Task:
    """
    Một task trong MaTOP.
    Bounds của x:
      x[0]   ∈ [0, 1]           (biến position)
      x[1:]  ∈ [lb_main, ub_main] (biến decision)
    """
    task_id:      int
    dim:          int
    lb:           np.ndarray     # shape (dim,)
    ub:           np.ndarray     # shape (dim,)
    problem_id:   int

    _shift:       np.ndarray = field(default=None, repr=False)   # shape (dim-1,)
    _rotation:    np.ndarray = field(default=None, repr=False)   # shape (dim-1, dim-1)
    _g_func:      Callable   = field(default=None, repr=False)
    _f1_type:     str        = field(default='linear', repr=False)
    _h_type:      str        = field(default='circle', repr=False)
    _lb_dec:      np.ndarray = field(default=None, repr=False)   # lb của phần decision
    _ub_dec:      np.ndarray = field(default=None, repr=False)   # ub của phần decision

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Đánh giá 1 giải pháp x (shape: dim,) → [f1, f2]
        """
        return self.evaluate_batch(x.reshape(1, -1))[0]

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Đánh giá N giải pháp X (shape: N x dim) → (N, 2)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        N = X.shape[0]

        # Tách biến position (cột 0) và biến decision (cột 1..)
        x_pos = X[:, 0:1]        # (N, 1) — x[0] ∈ [0,1]
        x_dec = X[:, 1:]         # (N, dim-1)

        # Áp dụng rotation + shift lên phần decision
        # z = R * (x_dec - shift)  — đúng theo MATLAB: (R * (x - shift)')'
        z = (self._rotation @ (x_dec - self._shift).T).T  # (N, dim-1)

        # Clamp về bounds
        z = np.clip(z, self._lb_dec, self._ub_dec)

        # Tính g-function
        g_val = self._g_func(z)  # (N,)

        # Problem loại ZDT (2, 3, 5) vs DTLZ (1, 4, 6)
        if self.problem_id in (2, 3, 5):
            # ZDT style
            f1 = eval_f1(x_pos, self._f1_type)     # (N,)
            g_val = g_val + 1
            h_val = eval_h(f1, g_val, self._h_type)
            f2 = g_val * h_val
        else:
            # DTLZ style
            g_val_shifted = g_val  # g không +1 ở đây
            f1 = (1 + g_val_shifted) * np.cos(x_pos[:, 0] * 0.5 * np.pi)
            f2 = (1 + g_val_shifted) * np.sin(x_pos[:, 0] * 0.5 * np.pi)

        return np.column_stack([f1, f2])  # (N, 2)

    def random_solution(self, rng=None) -> np.ndarray:
        """Tạo 1 giải pháp ngẫu nhiên hợp lệ trong bounds."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lb, self.ub)

    def random_population(self, n: int, rng=None) -> np.ndarray:
        """Tạo population N giải pháp ngẫu nhiên → (N, dim)"""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lb, self.ub, size=(n, self.dim))


# ──────────────────────────────────────────────────────────────
# Pareto front tham chiếu
# ──────────────────────────────────────────────────────────────

def get_pareto_front(problem_id: int, n_points: int = 1000) -> np.ndarray:
    """
    Trả về Pareto front tham chiếu (lý thuyết) cho mỗi MaTOP.
    Shape: (n_points, 2)
    """
    if problem_id in (1, 4, 6):  # circle
        theta = np.linspace(0, np.pi / 2, n_points)
        f1 = np.cos(theta)
        f2 = np.sin(theta)
    elif problem_id in (2, 3):   # concave
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
    elif problem_id == 5:         # convex
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 0.5
    return np.column_stack([f1, f2])


# ──────────────────────────────────────────────────────────────
# Lớp chính: CEC19_MaTMO
# ──────────────────────────────────────────────────────────────

class CEC19_MaTMO:
    """
    Benchmark CEC19 MaTMO — port chính xác từ MATLAB gốc.

    Args:
        problem_id : 1..6 (tương ứng CEC19_MaTMO1..6)
        data_dir   : đường dẫn đến thư mục chứa M/ và SVector/
        num_tasks  : số tasks (mặc định 50)

    Example:
        problem = CEC19_MaTMO(problem_id=1, data_dir='.')
        tasks = problem.tasks

        x = tasks[0].random_solution()
        f = tasks[0].evaluate(x)
        print(f"f1={f[0]:.4f}, f2={f[1]:.4f}")
    """

    def __init__(self, problem_id: int, data_dir: str = '.', num_tasks: int = 50):
        if problem_id not in range(1, 7):
            raise ValueError(f"problem_id phải từ 1 đến 6, nhận: {problem_id}")

        self.problem_id = problem_id
        self.num_tasks = num_tasks
        self.data_dir = data_dir

        dim, lb_main, ub_main, g_type, f1_type, h_type = PROBLEM_CONFIGS[problem_id]
        self.dim = dim

        # Load data files
        shifts, rotations = self._load_data(problem_id, num_tasks, data_dir)

        # Tạo tasks
        self.tasks: List[Task] = []
        for t in range(num_tasks):
            # Bounds: x[0] ∈ [0,1], x[1:] ∈ [lb_main, ub_main]
            lb = np.full(dim, lb_main, dtype=float)
            ub = np.full(dim, ub_main, dtype=float)
            lb[0] = 0.0
            ub[0] = 1.0

            lb_dec = np.full(dim - 1, lb_main, dtype=float)
            ub_dec = np.full(dim - 1, ub_main, dtype=float)

            task = Task(
                task_id=t + 1,
                dim=dim,
                lb=lb,
                ub=ub,
                problem_id=problem_id,
                _shift=shifts[t],
                _rotation=rotations[t],
                _g_func=G_FUNCTIONS[g_type],
                _f1_type=f1_type,
                _h_type=h_type,
                _lb_dec=lb_dec,
                _ub_dec=ub_dec,
            )
            self.tasks.append(task)

    @staticmethod
    def _load_data(problem_id: int, num_tasks: int, data_dir: str):
        """Đọc ShiftVector và RotationMatrix từ file .txt (giống readData_CEC19_MaTMO.m)"""
        shifts = []
        rotations = []
        for i in range(1, num_tasks + 1):
            s_path = os.path.join(data_dir, 'SVector', f'S{problem_id}', f'S{problem_id}_{i}.txt')
            m_path = os.path.join(data_dir, 'M',       f'M{problem_id}', f'M{problem_id}_{i}.txt')

            if not os.path.exists(s_path):
                raise FileNotFoundError(f"Không tìm thấy: {s_path}")
            if not os.path.exists(m_path):
                raise FileNotFoundError(f"Không tìm thấy: {m_path}")

            s = np.loadtxt(s_path)   # shape (dim-1,)
            m = np.loadtxt(m_path)   # shape (dim-1, dim-1)

            shifts.append(s)
            rotations.append(m)
        return shifts, rotations

    def get_pareto_front(self, n_points: int = 1000) -> np.ndarray:
        """Pareto front lý thuyết của MaTOP này. Shape: (n_points, 2)"""
        return get_pareto_front(self.problem_id, n_points)

    def __repr__(self):
        cfg = PROBLEM_CONFIGS[self.problem_id]
        return (f"CEC19_MaTMO{self.problem_id} | "
                f"tasks={self.num_tasks} | dim={self.dim} | "
                f"g={cfg[3]} | PF={cfg[5]}")


# ──────────────────────────────────────────────────────────────
# Demo nhanh
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Tự động tìm data_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir

    print("=" * 65)
    print("CEC19 MaTMO — Python Benchmark (port từ MATLAB gốc)")
    print("=" * 65)

    rng = np.random.default_rng(42)

    for pid in range(1, 7):
        try:
            prob = CEC19_MaTMO(problem_id=pid, data_dir=data_dir, num_tasks=50)
            print(f"\n{prob}")

            task = prob.tasks[0]
            print(f"  Bounds x[0]  : [{task.lb[0]}, {task.ub[0]}]  (position var)")
            print(f"  Bounds x[1:] : [{task.lb[1]}, {task.ub[1]}]  (decision vars)")

            # Evaluate 1 solution
            x = task.random_solution(rng)
            f = task.evaluate(x)
            print(f"  Task 1 sample: f1={f[0]:.5f}, f2={f[1]:.5f}")

            # Evaluate batch
            X = task.random_population(100, rng)
            F = task.evaluate_batch(X)
            print(f"  Batch (100) : f1∈[{F[:,0].min():.3f}, {F[:,0].max():.3f}]  "
                  f"f2∈[{F[:,1].min():.3f}, {F[:,1].max():.3f}]")

        except FileNotFoundError as e:
            print(f"\nMaTMO{pid}: ❌ {e}")

    print("\n" + "=" * 65)
    print("Cách tích hợp vào thuật toán của bạn:")
    print("=" * 65)
    print("""
  from cec19_matmo import CEC19_MaTMO
  import numpy as np

  # Load bài toán (cần thư mục M/ và SVector/ ở cùng chỗ)
  prob = CEC19_MaTMO(problem_id=1, data_dir='/path/to/CEC19_benchmark')
  tasks = prob.tasks  # list 50 Task

  for task in tasks:
      # Tạo population N=100 cá thể
      X = task.random_population(100)   # shape (100, dim)
      
      # Evaluate toàn bộ population
      F = task.evaluate_batch(X)        # shape (100, 2)
      
      # Lấy Pareto front tham chiếu để tính IGD, HV,...
      pf = prob.get_pareto_front()      # shape (1000, 2)
""")
