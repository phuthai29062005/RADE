import matplotlib.pyplot as plt
import ast
import numpy as np

def load_data(file_name):
    """Đọc và xử lý dữ liệu từ tệp tin"""
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    data.append(ast.literal_eval(line))
                except:
                    continue
    return np.array(data)

def plot_group(matea, rade, indices, filename):
    """Vẽ một nhóm các đồ thị vào một file ảnh"""
    n = len(indices)
    # Tính toán số hàng và cột (tối đa 2 cột)
    rows = (n + 1) // 2
    cols = 2 if n > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).flatten() # Đảm bảo axes là mảng 1 chiều
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(matea[:, idx], label='MaTEA', color='blue', linewidth=1.2)
        ax.plot(rade[:, idx], label='RADE', color='red', linestyle='--', linewidth=1.2)
        ax.set_title(f'Benchmark Function {idx + 1}', fontsize=12)
        ax.set_yscale('log') # Thang đo log để quan sát rõ hơn
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Value')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.3)
        
    # Ẩn các trục thừa nếu số lượng đồ thị lẻ
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Đã lưu: {filename}")
    plt.close()

# 1. Tải dữ liệu
matea_data = load_data('MaTEA.txt')
rade_data = load_data('RADE.txt')

# 2. Vẽ thành 3 ảnh theo yêu cầu
# Ảnh 1: 3 đồ thị đầu (0, 1, 2)
plot_group(matea_data, rade_data, range(0, 3), 'plots_part1.png')

# Ảnh 2: 3 đồ thị tiếp theo (3, 4, 5)
plot_group(matea_data, rade_data, range(3, 6), 'plots_part2.png')

# Ảnh 3: 4 đồ thị cuối (6, 7, 8, 9)
plot_group(matea_data, rade_data, range(6, 10), 'plots_part3.png')