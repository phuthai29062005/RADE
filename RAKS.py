import numpy as np

from benchmark import *
from KLD import *

STAGNATION = 1
EXPLOITATION = 2
NORMAL = 0

def RAKS(task, ES):
    num_task = len(ES) - 1
    selected_source_tasks = [None] * (num_task + 1)

    for i in range(1, num_task + 1):
        if ES[i] == NORMAL:
            selected_source_tasks[i] = i
            continue

        min_kld = float('inf')
        best_source = i
        avg_task = np.mean(task[i], axis=0)

        for j in range(1, num_task + 1):
            if j == i:
                continue

            if ES[i] == STAGNATION and ES[j] == STAGNATION:
                continue

            if ES[i] == EXPLOITATION and ES[j] != EXPLOITATION:
                continue

            avg_sources = np.mean(task[j], axis=0)
            shifted_task_j = task[j] - avg_sources + avg_task
            current_kld = KLD(task[i], shifted_task_j)

            if current_kld < min_kld:
                min_kld = current_kld
                best_source = j

        selected_source_tasks[i] = best_source

    return selected_source_tasks