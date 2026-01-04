import numpy as np
import random



def KLD(task0, task1):
    
    D0 = task0.shape[1]
    D1 = task1.shape[1]
    D = min(D0, D1)
    
    data0 = task0[:, :D]
    data1 = task1[:, :D]
    
    mean_0 = np.mean(data0, axis=0)
    mean_1 = np.mean(data1, axis=0)
    
    sigma_0 = np.cov(data0, rowvar = False) + np.eye(D) * 1e-4
    sigma_1 = np.cov(data1, rowvar = False) + np.eye(D) * 1e-4
    
    #Tính nghịch đảo Sigma_1 (Sigma_1^-1)
    sigma_1_inv = np.linalg.inv(sigma_1)
    
    # Thành phần A: tr(Sigma_1^-1 * Sigma_0)
    term_trace = np.trace(sigma_1_inv @ sigma_0)
    
    # Thành phần B: (mu_1 - mu_0)^T * Sigma_1^-1 * (mu_1 - mu_0)
    
    diff_mean = mean_1 - mean_0
    term_mahalanobis = diff_mean.T @ sigma_1_inv @ diff_mean
    
    # Thành phần C: ln(det(Sigma_1) / det(Sigma_0))
    # = ln(det(Sigma_1)) - ln(det(Sigma_0))
    # Dùng slogdet để tính log định thức an toàn, tránh tràn số (overflow)
    sign1, logdet1 = np.linalg.slogdet(sigma_1)
    if sign1 <= 0: logdet1 = -500 
    sign0, logdet0 = np.linalg.slogdet(sigma_0)
    if sign0 <= 0: logdet0 = -500
    term_logdet = logdet1 - logdet0
    
    kld = term_trace + term_mahalanobis - D + term_logdet
    return max(0.0, kld)


    
    
    