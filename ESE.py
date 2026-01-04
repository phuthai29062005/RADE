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
        avg_current = np.mean(task[i], axis=0)
        
        r[gen] = np.sum(Nice_gen[i][gen - delta_g : gen]) / (delta_g * population)
        
        distance = np.linalg.norm(task[i] - avg_current, axis = 1)
        T[gen] = np.mean(distance)
        
        if r[gen] <= Or: #STAGNATION
            ES[i] = 1
        elif T[gen] <= T[gen - delta_g] * Od: #EXPLOITATION
            ES[i] = 2
        
    return ES
    
       
            
        
      
        
        