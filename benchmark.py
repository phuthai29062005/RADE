import numpy as np

# --- CẤU HÌNH THÔNG SỐ TỪ BẢNG I (TABLE I) ---
def get_task_info(task_id):
    """
    Lấy thông tin cấu hình cho 10 Task theo Chen et al. (2020).
    Tham khảo: Table I, Page 8 của bài báo.
    """
    # Mặc định D=50 cho hầu hết các task
    dim = 50
    shift = np.zeros(dim)
    bounds = [-100, 100] # Default range

    if task_id == "T1": # Sphere (E-task)
        bounds = [-100, 100]
        shift[:] = 0
        
    elif task_id == "T2": # Sphere (E-task)
        bounds = [-100, 100]
        shift[:] = 80
        
    elif task_id == "T3": # Sphere (E-task)
        bounds = [-100, 100]
        shift[:] = -80
        
    elif task_id == "T4": # Weierstrass (E-task) -> D=25
        dim = 25
        bounds = [-0.5, 0.5]
        shift = np.zeros(dim)
        shift[:] = -0.4
        
    elif task_id == "T5": # Rosenbrock (C-task)
        bounds = [-50, 50]
        shift[:] = 0
        
    elif task_id == "T6": # Ackley (C-task)
        bounds = [-50, 50]
        shift[:] = 40
        
    elif task_id == "T7": # Weierstrass (C-task)
        bounds = [-0.5, 0.5]
        shift[:] = -0.4
        
    elif task_id == "T8": # Schwefel (C-task)
        bounds = [-500, 500]
        shift[:] = 420.9687
        
    elif task_id == "T9": # Griewank (C-task)
        bounds = [-100, 100]
        # Shift vector: Nửa đầu -80, Nửa sau 80
        half = dim // 2
        shift[:half] = -80
        shift[half:] = 80
        
    elif task_id == "T10": # Rastrigin (C-task)
        bounds = [-50, 50]
        # Shift vector: Nửa đầu 40, Nửa sau -40
        half = dim // 2
        shift[:half] = 40
        shift[half:] = -40
        
    else:
        return None, None, None

    return shift, dim, bounds

# --- CÁC HÀM MỤC TIÊU (EQ. 13 - 19) ---

def sphere(x, shift):
    """Eq. 13: Sphere Function"""
    z = x - shift
    return np.sum(z**2)

def weierstrass(x, shift, a=0.5, b=3, k_max=20):
    """Eq. 14: Weierstrass Function"""
    z = x - shift
    D = len(x)
    
    # Pre-compute k terms
    k = np.arange(0, k_max + 1)
    a_k = a**k
    b_k = b**k
    
    # Term 1: Sum over D and K
    term1 = 0
    for i in range(D):
        term1 += np.sum(a_k * np.cos(2 * np.pi * b_k * (z[i] + 0.5)))
        
    # Term 2: Constant
    term2 = D * np.sum(a_k * np.cos(2 * np.pi * b_k * 0.5))
    
    return term1 - term2

def rosenbrock(x, shift):
    """Eq. 15: Rosenbrock Function"""
    # Lưu ý: Rosenbrock thường ít khi shift trực tiếp kiểu z = x - shift rồi mới tính
    # Tuy nhiên bài báo ghi z = x - O (đoạn dưới bảng I).
    z = x - shift
    D = len(x)
    val = 0
    for i in range(D - 1):
        val += 100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
    return val

def ackley(x, shift):
    """Eq. 16: Ackley Function"""
    z = x - shift
    D = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / D))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * z)) / D)
    return term1 + term2 + 20 + np.e

def schwefel(x, shift):
    """Eq. 17: Schwefel Function"""
    # Bài báo ghi z = x - O.
    # Hàm gốc Schwefel định nghĩa trên [-500, 500].
    z = x - shift 
    D = len(x)
    
    # Xử lý biên an toàn để tránh lỗi sqrt số âm nếu z trôi ra quá xa (dù bounds đã chặn)
    # Trong các benchmark chuẩn, thường dùng z_i * sin(sqrt(abs(z_i)))
    sum_val = np.sum(z * np.sin(np.sqrt(np.abs(z))))
    
    return 418.9829 * D - sum_val

def griewank(x, shift):
    """Eq. 18: Griewank Function"""
    z = x - shift
    D = len(x)
    
    sum_sq = np.sum(z**2) / 4000
    
    # Prod cos(z_i / sqrt(i)) . Lưu ý index i chạy từ 1 đến D
    indices = np.arange(1, D + 1)
    prod_cos = np.prod(np.cos(z / np.sqrt(indices)))
    
    return 1 + sum_sq - prod_cos

def rastrigin(x, shift):
    """Eq. 19: Rastrigin Function"""
    z = x - shift
    D = len(x)
    return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

# --- WRAPPER CHÍNH ---

def calculate_objective_function(task_id, x, shift):
    """
    Hàm gọi chính để tính Fitness.
    Input:
        task_id: "T1", "T2", ...
        x: vector biến số (đã denormalize về miền thực)
        shift: vector dịch chuyển (lấy từ get_task_info)
    """
    if task_id in ["T1", "T2", "T3"]:
        return sphere(x, shift)
    elif task_id == "T4":
        return weierstrass(x, shift)
    elif task_id == "T5":
        return rosenbrock(x, shift)
    elif task_id == "T6":
        return ackley(x, shift)
    elif task_id == "T7":
        return weierstrass(x, shift)
    elif task_id == "T8":
        return schwefel(x, shift)
    elif task_id == "T9":
        return griewank(x, shift)
    elif task_id == "T10":
        return rastrigin(x, shift)
    else:
        return 1e9 # Giá trị phạt nếu task ID sai