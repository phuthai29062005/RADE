## đánh giá hiệu năng của các nhiệm vụ
import numpy as np
from benchmark import *

def ESE(task, gen, Or, Od, Nice_gen, r, T):
    
    num_task = 10
    delta_g = 10
    population = 100
    
    ES = np.zeros(num_task + 1)
    for i in range(1, num_task + 1):
        
        ## toạ độ trung bình hiện tại
        _, dim, _ = get_task_info(f"T{i}")
        task_coords = task[i][:, :dim]
        avg_current = np.mean(task_coords, axis=0)
        distance = np.linalg.norm(task_coords - avg_current, axis=1)
        T[gen][i] = np.mean(distance)
        
        for j in range(gen - delta_g, gen):
            r[gen][i] += Nice_gen[j][i]
        r[gen][i] /= population
        
        
        if r[gen][i] <= Or: #STAGNATION
            ES[i] = 1
        elif T[gen][i] <= T[gen - delta_g][i] * Od: #EXPLOITATION
            ES[i] = 2
    
    return ES
    
       
            
        
      
        
        