import numpy as np

from benchmark import *
from DE import *

STAGNATION = 1
EXPLOITATION = 2
NORMAL = 0

def RAKT(gen, task, ES, fitness_arr, selected_task, alpha, F, CR, population, dim_max):

    nice_gen = np.zeros(num_task + 1)
    num_task = 10
    for i in range(num_task + 1):
        
        if ES[i] == NORMAL:
            task[i], fitness_arr[i], nice_gen[i] = DE(i, task[i], fitness_arr[i], F, CR, population)
        if ES[i] == STAGNATION:

            task[i], fitness_arr[i], nice_gen[i] = DE_STAG(i, task, selected_task[i], fitness_arr, alpha, F, CR, population)

        if ES[i] == EXPLOITATION:
            task[i], fitness_arr[i], nice_gen[i] = DE_EXP(i, task, selected_task[i], fitness_arr, alpha, F, CR, population)
    
    
    return task, fitness_arr, nice_gen