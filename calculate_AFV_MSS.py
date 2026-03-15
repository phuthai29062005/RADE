import numpy as np
import re

def calculate_paper_table():
    output_file = "RADE_Table_Results.txt"
    # Giả sử bạn chưa có dữ liệu các thuật toán khác, ta tính AFV trước.
    # Nếu có f_mean và f_std của các thuật toán khác, bạn hãy thay vào mảng tương ứng.
    
    header = f"{'Problem':<12} | {'AFV (Mean)':<15} | {'MSS (Mean)':<15}"
    print(header)
    print("-" * 46)
    
    with open(output_file, "w") as f_out:
        f_out.write(header + "\n")
        f_out.write("-" * 46 + "\n")
        
        for i in range(1, 7):
            file_name = f"RADE_Matrix_{i}.txt"
            try:
                # 1. Đọc dữ liệu từ file Matrix (50 tasks x 10 seeds)
                with open(file_name, 'r') as f:
                    content = f.read()
                    numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+", content)
                    data = np.array(numbers, dtype=float)
                
                matrix = data.reshape(50, 10) # 50 hàng (tasks), 10 cột (seeds)
                
                # 2. Tính AFV cho từng seed (Trung bình của 50 tasks mỗi cột)
                # AFV_seed_j = (f1,j + f2,j + ... + f50,j) / 50
                afv_seeds = np.mean(matrix, axis=0)
                mean_afv = np.mean(afv_seeds)
                
                # 3. Tính MSS (Tạm thời tính dựa trên 10 seed này làm mốc)
                # Trong thực tế, bạn cần f_mean, f_std của tất cả các thuật toán.
                # MSS_seed_j = Mean over tasks of ((fi,j - f_mean_i) / f_std_i)
                task_means = np.mean(matrix, axis=1, keepdims=True)
                task_stds = np.std(matrix, axis=1, keepdims=True)
                # Tránh chia cho 0 nếu std = 0
                task_stds[task_stds == 0] = 1.0
                
                normalized_matrix = (matrix - task_means) / task_stds
                mss_seeds = np.mean(normalized_matrix, axis=0)
                mean_mss = np.mean(mss_seeds)
                
                # 4. In và ghi file
                line = f"CEC19-P{i:<5} | {mean_afv:<15.6e} | {mean_mss:<15.6e}"
                print(line)
                f_out.write(line + "\n")
                
            except Exception as e:
                print(f"CEC19-P{i:<5} | Lỗi: File không tồn tại hoặc sai định dạng")

    print(f"\nĐã tạo file bảng kết quả: {output_file}")

if __name__ == "__main__":
    calculate_paper_table()