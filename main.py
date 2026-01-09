import numpy as np
import time
from benchmark import *
from ESE import ESE
from RAKS import RAKS
from RAKT import RAKT

def log_to_file(fitness):
    with open("RADE.txt", "a") as f:
        f.write(f"{fitness}\n")

def log_to_file_ES(ES, selected_task, r, T, T_prev):
    with open("RADE_ES.txt", "a") as f:
        f.write(f"ES {ES}\n")
        f.write(f"Selected Task {selected_task}\n")
        f.write(f"R {r}\n")
        f.write(f"T {T}\n")
        f.write(f"T_prev {T_prev}\n")
        f.write("\n")
    
def clear_old_logs():
    with open("RADE.txt", "w") as f:
        f.write("Fitness\n")

def clear_old_logs_ES():
    with open("RADE_ES.txt", "w") as f:
        f.write("ES\n")

        
def calculate_fitness(task, i_idx):
    real_gen = task.copy()
    shift, dim, bounds = get_task_info(f"T{i_idx}")
    real_gen = bounds[0] + real_gen[:dim] * (bounds[1] - bounds[0])
    fitness = calculate_objective_function(f"T{i_idx}", real_gen, shift)
    return fitness

def RADE():
    
    clear_old_logs()
    clear_old_logs_ES()
    generation = 5000
    population = 100
    seed = 10
    F = 0.7 # scaling factor
    CR = 0.5 # crossover rate
    Or = 0.1 # threshold for stagnation detection
    Od = 0.9 # threshold for diversity assessment
    alpha = 0.2 # knowledge transfer balancing coefficient
    delta_g = 10 # window
    num_task = 10
    avg_fitness = np.zeros((generation + 1, num_task + 1))
    
    dim_max = 0
    for i in range(1, num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        if dim > dim_max:
            dim_max = dim
        
    for test in range(seed):
        
        print(f"Starting Run {test + 1}/{seed}...")
        task = [[] for _ in range(num_task + 1)]
        fitness_arr = [np.zeros(population) for _ in range(num_task + 1)]
        Nice_gen = np.zeros((generation + 1, num_task + 1))
        r = np.zeros((generation + 1, num_task + 1))
        T = np.zeros((generation + 1, num_task + 1))
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
            
            for t in range(1, num_task + 1):
                best_fitness = np.min(fitness_arr[t])
                avg_fitness[i][t] += best_fitness
                
            if i % 100 == 0:     
                print(f"Generation {i} completed.")
                
            if i % delta_g == 1 and i > 10:
                ## đánh giá hiệu năng
                ES = ESE(task, i, Or, Od, Nice_gen, r, T)
                selected_task = RAKS(task, ES)
                x = T[i - delta_g][1:] * Od
                log_to_file_ES(ES[1:].tolist(), selected_task[1:], r[i][1:].tolist(), T[i][1:].tolist(), x.tolist())
            
             
            task, fitness_arr, Nice_gen[i] = RAKT(i, task, ES, fitness_arr, selected_task, alpha, F, CR, population, dim_max)
            for t in range(1, num_task + 1):
                avg_current = np.mean(task[t], axis=0)
                distance = np.linalg.norm(task[t] - avg_current, axis = 1)
                T[i][t] = np.mean(distance)
            #print(ES) 
            #print(Nice_gen[i])
    
    avg_fitness_matrix = avg_fitness/ seed

    for i in range(1, generation + 1):
        log_to_file(avg_fitness_matrix[i, 1:].tolist())
        

if __name__ == "__main__":
    start_time = time.time()
    RADE()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # Đổi sang phút
    print(f"Thuật toán chạy trong {elapsed_time:.2f} phút")
