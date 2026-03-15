"""
cec19_matso.py
──────────────────────────────────────────────────────────────
CEC19 Many-task Single-objective Optimization (MaTSO) Benchmark
Port từ MATLAB: benchmark_CEC19_MaTSO.m + getFun

6 MaTSOPs, mỗi bài 50 tasks, dim=50, single-objective.

Problem configs:
  P1: Rosenbrock,   bounds=[-50,   50]
  P2: Ackley,       bounds=[-50,   50]
  P3: Rastrigin,    bounds=[-50,   50]
  P4: Griewank,     bounds=[-100, 100]
  P5: Weierstrass,  bounds=[-0.5, 0.5]
  P6: Schwefel,     bounds=[-500, 500]

Cách tính (đúng với MATLAB MToP):
  z = M * (x - o)'   (M: rotation 50x50, o: shift 50-dim)
  f = base_function(z)
"""

import os
import numpy as np
import scipy.io as sio

# Cấu hình 6 MaTSOPs
PROBLEM_CONFIGS = {
    1: ('Rosenbrock',   -50.0,   50.0),
    2: ('Ackley',       -50.0,   50.0),
    3: ('Rastrigin',    -50.0,   50.0),
    4: ('Griewank',    -100.0,  100.0),
    5: ('Weierstrass',  -0.5,    0.5),
    6: ('Schwefel',    -500.0,  500.0),
}


# Hàm base (port từ MATLAB MToP)
def _rosenbrock(z):
    d = len(z)
    s = 0.0
    for i in range(d - 1):
        s += 100.0 * (z[i + 1] - z[i] ** 2) ** 2 + (z[i] - 1.0) ** 2
    return s

def _ackley(z):
    d = len(z)
    s1 = np.sqrt(np.sum(z ** 2) / d)
    s2 = np.sum(np.cos(2.0 * np.pi * z)) / d
    return -20.0 * np.exp(-0.2 * s1) - np.exp(s2) + 20.0 + np.e

def _rastrigin(z):
    d = len(z)
    return np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z)) + 10.0 * d

def _griewank(z):
    d = len(z)
    s = np.sum(z ** 2) / 4000.0
    p = np.prod(np.cos(z / np.sqrt(np.arange(1, d + 1))))
    return 1.0 + s - p

def _weierstrass(z, a=0.5, b=3.0, k_max=20):
    d = len(z)
    k  = np.arange(0, k_max + 1, dtype=float)
    ak = a ** k
    bk = b ** k
    term1 = np.sum(ak * np.cos(2.0 * np.pi * bk * (z[:, None] + 0.5)))
    term2 = d * np.sum(ak * np.cos(np.pi * bk))
    return term1 - term2

def _schwefel(z):
    d = len(z)
    return 418.9829 * d - np.sum(z * np.sin(np.sqrt(np.abs(z))))

BASE_FUNCTIONS = {
    'Rosenbrock':  _rosenbrock,
    'Ackley':      _ackley,
    'Rastrigin':   _rastrigin,
    'Griewank':    _griewank,
    'Weierstrass': _weierstrass,
    'Schwefel':    _schwefel,
}


class Task:
    def __init__(self, task_id, dim, lb, ub, rotation, shift, func_name):
        self.task_id   = task_id
        self.dim       = dim
        self.lb        = lb    # scalar
        self.ub        = ub    # scalar
        self._rotation = np.asarray(rotation, dtype=float)  # (dim, dim)
        self._shift    = np.asarray(shift,    dtype=float)  # (dim,)
        self._func     = BASE_FUNCTIONS[func_name]
        self._fname    = func_name

    def evaluate(self, x):
        """x: (dim,) thực trong [lb, ub] → scalar fitness"""
        x = np.asarray(x, dtype=float)
        z = self._rotation @ (x - self._shift)
        return float(self._func(z))

    def evaluate_batch(self, X):
        """X: (N, dim) → (N,)"""
        return np.array([self.evaluate(row) for row in X])

    def __repr__(self):
        return f"Task{self.task_id}({self._fname}, dim={self.dim}, [{self.lb},{self.ub}])"


class CEC19_MaTSO:
    """
    CEC19 Many-task Single-objective Optimization benchmark.

    Args:
        problem_id : 1..6
        data_dir   : thư mục chứa Tasks/ (hoặc GoTask*.mat trực tiếp)
        num_tasks  : số tasks cần dùng (tối đa 50)
    """
    def __init__(self, problem_id, data_dir='.', num_tasks=50):
        if problem_id not in PROBLEM_CONFIGS:
            raise ValueError(f"problem_id phải 1..6, nhận: {problem_id}")

        self.problem_id = problem_id
        self.num_tasks  = num_tasks
        func_name, lb, ub = PROBLEM_CONFIGS[problem_id]
        self.func_name = func_name
        self.lb, self.ub = lb, ub
        self.dim = 50

        # Tìm file .mat (Tasks/ subfolder hoặc thư mục gốc)
        tasks_dir = os.path.join(data_dir, 'Tasks')
        go_path   = os.path.join(tasks_dir, f'GoTask{problem_id}.mat')
        if not os.path.exists(go_path):
            tasks_dir = data_dir
            go_path   = os.path.join(tasks_dir, f'GoTask{problem_id}.mat')
        rot_path = os.path.join(tasks_dir, f'RotationTask{problem_id}.mat')

        if not os.path.exists(go_path):
            raise FileNotFoundError(f"Không tìm thấy: {go_path}")
        if not os.path.exists(rot_path):
            raise FileNotFoundError(f"Không tìm thấy: {rot_path}")

        go_data  = sio.loadmat(go_path)
        rot_data = sio.loadmat(rot_path)

        # GoTask{id}:       shape (50, 50) — hàng i là shift của task i+1
        # RotationTask{id}: cell (1, 50)   — rot[0,i] là (50,50) của task i+1
        shifts    = go_data[f'GoTask{problem_id}']
        rot_cells = rot_data[f'RotationTask{problem_id}']

        self.tasks = []
        for t in range(num_tasks):
            self.tasks.append(Task(
                task_id   = t + 1,
                dim       = self.dim,
                lb        = lb,
                ub        = ub,
                rotation  = rot_cells[0, t],
                shift     = shifts[t],
                func_name = func_name,
            ))

    def __repr__(self):
        return (f"CEC19_MaTSO{self.problem_id} | tasks={self.num_tasks} | "
                f"dim={self.dim} | func={self.func_name} | bounds=[{self.lb},{self.ub}]")
