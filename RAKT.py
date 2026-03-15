import numpy as np

from benchmark import *
from DE import *

STAGNATION = 1
EXPLOITATION = 2
NORMAL = 0

def RAKT(gen, task, ES, fitness_arr, selected_task, alpha, F, CR, population, dim_max):

    num_task = len(ES) - 1  # tự động theo số task thực tế
    nice_gen = np.zeros(num_task + 1)
    
    for i in range(1, num_task + 1):
        
        if ES[i] == NORMAL:
            task[i], fitness_arr[i], nice_gen[i] = DE(i, task[i], fitness_arr[i], F, CR, population, dim_max)
        if ES[i] == STAGNATION:
            if selected_task[i] == i:
                task[i], fitness_arr[i], nice_gen[i] = DE(i, task[i], fitness_arr[i], F, CR, population, dim_max)
            else:
                task[i], fitness_arr[i], nice_gen[i] = DE_STAG(i, task, selected_task[i], fitness_arr, alpha, F, CR, population, dim_max)

        if ES[i] == EXPLOITATION:
            if selected_task[i] == i:
                task[i], fitness_arr[i], nice_gen[i] = DE(i, task[i], fitness_arr[i], F, CR, population, dim_max)
            else:
                task[i], fitness_arr[i], nice_gen[i] = DE_EXP(i, task, selected_task[i], fitness_arr, alpha, F, CR, population, dim_max)
    
    return task, fitness_arr, nice_gen