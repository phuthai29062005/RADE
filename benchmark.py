"""
benchmark.py  —  CEC19 MaTSO Benchmark Adapter
Giữ nguyên interface cho main.py:
    get_task_info(task_id_str)                           → (shift, dim, bounds)
    calculate_objective_function(task_id_str, x_real, shift) → scalar fitness

Để đổi problem (P1..P6), sửa ACTIVE_PROBLEM_ID bên dưới.
"""

import os
import numpy as np
from cec19_matso import CEC19_MaTSO

# ─── Cấu hình ───────────────────────────────────────────────
ACTIVE_PROBLEM_ID = 6   # Đổi thành 1..6 để chọn CEC19-MaTSO P1..P6
NUM_TASKS         = 50  # 50 tasks theo chuẩn CEC19
# ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'cec19_single', 'MTO-Platform', 'MTO', 'Problems', 'Multi-task', 'CEC19-MaTSO'
)

_problem = CEC19_MaTSO(
    problem_id = ACTIVE_PROBLEM_ID,
    data_dir   = DATA_DIR,
    num_tasks  = NUM_TASKS,
)
_tasks = _problem.tasks  # list Task, index 0..49


def get_task_info(task_id_str: str):
    """
    task_id_str: "T1".."T50"
    Trả về: (shift=None, dim, [lb, ub])
    """
    idx  = int(task_id_str[1:]) - 1
    task = _tasks[idx]
    return None, task.dim, [task.lb, task.ub]


def calculate_objective_function(task_id_str: str, x_real, shift):
    """
    task_id_str : "T1".."T50"
    x_real      : vector đã decode về bounds thực, shape (dim,)
    shift       : bỏ qua (CEC19 tích hợp sẵn rotation+shift bên trong)
    Trả về      : scalar fitness
    """
    idx  = int(task_id_str[1:]) - 1
    task = _tasks[idx]
    x    = np.asarray(x_real, dtype=float).flatten()[:task.dim]
    return task.evaluate(x)