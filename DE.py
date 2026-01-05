import numpy as np
from benchmark import *

def DE_rand_1(task, target_idx, F, population):
    
    u1 = np.random.randint(0, population)
    u2 = np.random.randint(0, population)
    u3 = np.random.randint(0, population)
    
    while(u1 == target_idx):
        u1 = np.random.randint(0, population)
        
    while(u2 == target_idx or u2 == u1):
        u2 = np.random.randint(0, population)

    while(u3 == target_idx or u3 == u1 or u3 == u2):
        u3 = np.random.randint(0, population)
        
    child = task[u1] + F * (task[u2] - task[u3])
    child = np.clip(child, 0, 1)
    
    return child

def crossover(trial, child, j_rand, CR):
    
    for j in range(len(child)):
        rate = np.random.uniform(0, 1)
        if rate <= CR or j == j_rand:
            trial[j] = child[j]
    return trial

def get_mean(task0, task1):

    union_set = np.concatenate((task0, task1), axis=0)
    mean_vector = np.mean(union_set, axis=0)
    return mean_vector

def cauchy(child, dim):
    
    perturbation = np.random.standard_cauchy(dim) * 0.1
    new_child = child + perturbation
    new_child = np.clip(new_child, 0, 1)
    return new_child

def DE(x, task, fitness_arr, F, CR, population, dim_max):
    
    new_child = []
    shift, dim, bounds = get_task_info(f"T{x}")
    
    for i in range(population):
        new_child.append(((task[i], fitness_arr[i]), 0))
        
    for i in range(population):
        
        child = DE_rand_1(task, i, F, population)
        trial = task[i].copy()
        j_rand = np.random.randint(0, dim)
        
        trial = crossover(trial, child, j_rand, CR)
        real_gen = bounds[0] + trial[:dim] * (bounds[1] - bounds[0])
        fitness = calculate_objective_function(f"T{x}", real_gen, shift)
        new_child.append(((trial, fitness), 1))
        
    
    new_child.sort(key = lambda x : x[0][1])
    
    survivors = new_child[:population]
    
    # 4. Tách dữ liệu để trả về
    new_task = np.array([item[0][0] for item in survivors])
    new_fitness = np.array([item[0][1] for item in survivors])
    
    # 5. Tính nice_gen (N(gr)) - số lượng con (flag=1) sống sót [cite: 270, 271]
    nice_gen = sum(item[1] for item in survivors)
    
    return new_task, new_fitness, nice_gen

def DE_STAG(x, task, selected_task, fitness_arr, alpha, F, CR, population, dim_max):
    
    new_child = []
    mean_x = np.mean(task[x], axis=0)
    mean_s = np.mean(task[selected_task], axis=0)
    shifted_task_s = task[selected_task] - mean_s + mean_x
    
    mean_u = get_mean(shifted_task_s, task[x])
    rmp = alpha + 0.7
    
    shift, dim, bounds = get_task_info(f"T{x}")
    for i in range(population):
        new_child.append(((task[x][i], fitness_arr[x][i]), 0))
    
    for i in range(population):
        
        rate = np.random.uniform(0, 1)
        if rate < rmp:
        
            child = cauchy(mean_u, dim_max)
            real_gen = bounds[0] + child[:dim] * (bounds[1] - bounds[0])
            fitness = calculate_objective_function(f"T{x}", real_gen, shift)
            new_child.append(((child, fitness), 1))
        else:
            
            child = DE_rand_1(task[x], i, F, population)
            trial = task[x][i].copy()
            j_rand = np.random.randint(0, dim)
            
            trial = crossover(trial, child, j_rand, CR)
            real_gen = bounds[0] + trial[:dim] * (bounds[1] - bounds[0])
            fitness = calculate_objective_function(f"T{x}", real_gen, shift)
            
            new_child.append(((trial, fitness), 1))



    new_child.sort(key = lambda x : x[0][1])

    survivors = new_child[:population]
    
    # 4. Tách dữ liệu để trả về
    new_task = np.array([item[0][0] for item in survivors])
    new_fitness = np.array([item[0][1] for item in survivors])
    
    # 5. Tính nice_gen (N(gr)) - số lượng con (flag=1) sống sót [cite: 270, 271]
    nice_gen = sum(item[1] for item in survivors)
    
    return new_task, new_fitness, nice_gen                   

def DE_EXP(x, task, selected_task, fitness_arr, alpha, F, CR, population, dim_max):
    
    new_child = []
    mean_x = np.mean(task[x], axis=0)
    mean_s = np.mean(task[selected_task], axis=0)
    shifted_task_s = task[selected_task] - mean_s + mean_x
    
    union_set = np.concatenate((shifted_task_s, task[x]), axis=0)
    rmp = 0.7 - alpha
    
    shift, dim, bounds = get_task_info(f"T{x}")
    for i in range(population):
        new_child.append(((task[x][i], fitness_arr[x][i]), 0))
    
    best_idx = np.argmin(fitness_arr[x])
    x_best = task[x][best_idx]
    
    for i in range(population):
        
        rate = np.random.uniform(0, 1)
        if rate < rmp:
        
            # --- Chuyển giao liên tác vụ: DE/current-to-best/1 --- 
            # Chọn r1, r2 từ tập hợp gộp (union_set) [cite: 402]
            r1_idx, r2_idx = np.random.choice(len(union_set), 2, replace=False)
            xr1, xr2 = union_set[r1_idx], union_set[r2_idx]
            
            # Đột biến theo công thức (15) 
            child = task[x][i] + F * (x_best - task[x][i]) + F * (xr1 - xr2)
            child = np.clip(child, 0, 1)
            
            # Crossover [cite: 139]
            trial = task[x][i].copy()
            j_rand = np.random.randint(0, dim)
            trial = crossover(trial, child, j_rand, CR)
            
            real_gen = bounds[0] + trial[:dim] * (bounds[1] - bounds[0])
            fitness = calculate_objective_function(f"T{x}", real_gen, shift)
            new_child.append(((trial, fitness), 1))
        else:
            
            child = DE_rand_1(task[x], i, F, population)
            trial = task[x][i].copy()
            j_rand = np.random.randint(0, dim)
            
            trial = crossover(trial, child, j_rand, CR)
            real_gen = bounds[0] + trial[:dim] * (bounds[1] - bounds[0])
            fitness = calculate_objective_function(f"T{x}", real_gen, shift)
            
            new_child.append(((trial, fitness), 1))



    new_child.sort(key = lambda x : x[0][1])

    survivors = new_child[:population]
    
    # 4. Tách dữ liệu để trả về
    new_task = np.array([item[0][0] for item in survivors])
    new_fitness = np.array([item[0][1] for item in survivors])
    
    # 5. Tính nice_gen (N(gr)) - số lượng con (flag=1) sống sót [cite: 270, 271]
    nice_gen = sum(item[1] for item in survivors)
    
    return new_task, new_fitness, nice_gen                   