import numpy as np


from benchmark import *
from ESE import ESE
from RAKS import RAKS
from RAKT import RAKT

def calculate_fitness(task, i_idx):
    real_gen = task.copy()
    shift, dim, bounds = get_task_info(f"T{i_idx}")
    real_gen = bounds[0] + real_gen[:dim] * (bounds[1] - bounds[0])
    fitness = calculate_objective_function(f"T{i_idx}", real_gen, shift)
    return fitness

def RADE():
    
    generation = 5000000
    population = 100
    F = 0.7 # scaling factor
    CR = 0.5 # crossover rate
    Or = 0.1 # threshold for stagnation detection
    Od = 0.9 # threshold for diversity assessment
    alpha = 0.2 # knowledge transfer balancing coefficient
    delta_g = 10 # window
    num_task = 10
    
    dim_max = 0
    for i in range(1, num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        if dim > dim_max:
            dim_max = dim
        
    task = [[] for _ in range(num_task + 1)]
    fitness_arr = [np.zeros(population) for _ in range(num_task + 1)]
    Nice_gen = np.zeros((num_task + 1, generation + 1))
    r = np.zeros((num_task + 1, generation + 1))
    T = np.zeros((num_task + 1, generation + 1))
    ES = np.zeros(num_task + 1)
    selected_task = np.zeros(num_task + 1)
    
    for i in range(num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        task[i] = np.random.uniform(0, 1, (population, dim_max))
        if i > 0:
            for j in range(population):
                fitness = calculate_fitness(task[i][j], i)
                fitness_arr[i][j] = fitness
                
        
    for i in range(1, generation + 1):
        
        if i % delta_g == 0:
            ## đánh giá hiệu năng
            ES = ESE(task, i, Or, Od, Nice_gen, r, T)
            selected_task = RAKS(task, ES)
            
        task, fitness_arr, Nice_gen[i] = RAKT(i, task, ES, fitness_arr, selected_task, alpha, F, CR, population, dim_max)
    
    
    
