import time
import numpy as np

from benchmark import *
from ESE import ESE
from RAKS import RAKS
from RAKT import RAKT


def log_to_file(fitness_by_task):
    with open(f"RADE_{ACTIVE_PROBLEM_ID}.txt", "a") as f:
        f.write(f"{fitness_by_task}\n")


def calculate_fitness(individual, task_idx):
    """
    Decode cá thể từ unified space [0,1]^D về miền thật của task rồi tính fitness.
    Interface giữ đúng với benchmark.py hiện tại.
    """
    shift, dim, bounds = get_task_info(f"T{task_idx}")
    real_x = bounds[0] + individual[:dim] * (bounds[1] - bounds[0])
    return calculate_objective_function(f"T{task_idx}", real_x, shift)


def RADE():
    # ===== Tham số theo paper =====
    num_task = NUM_TASKS
    population = 100
    runs = 20
    F = 0.7
    CR = 0.5
    Or = 0.1       # Theta_r
    Od = 0.9       # Theta_d
    alpha = 0.2
    delta_g = 10
    max_fes = 5_000_000

    # Tính generation sao cho tổng số lần evaluate = max_fes,
    # có tính cả population khởi tạo.
    init_fes = num_task * population
    fes_per_generation = num_task * population
    remaining_fes = max_fes - init_fes
    if remaining_fes < 0 or remaining_fes % fes_per_generation != 0:
        raise ValueError("max_fes không tương thích với num_task và population hiện tại.")
    generation = remaining_fes // fes_per_generation

    # Dùng để lưu đường cong hội tụ trung bình theo run
    avg_fitness = np.zeros((generation + 1, num_task + 1))

    # Dùng để lưu best fitness cuối cùng của mỗi run trên từng task
    matrix_fitness = np.zeros((runs, num_task))

    # Tìm dim_max trong benchmark
    dim_max = 0
    for t in range(1, num_task + 1):
        _, dim, _ = get_task_info(f"T{t}")
        dim_max = max(dim_max, dim)

    print(
        f"CEC19-P{ACTIVE_PROBLEM_ID} | {num_task} tasks | dim={dim_max} | "
        f"{generation} gens | {runs} runs | maxFEs={max_fes}"
    )

    # Xóa log cũ để tránh append lẫn kết quả cũ
    open(f"RADE_{ACTIVE_PROBLEM_ID}.txt", "w").close()

    for run in range(runs):
        np.random.seed(run + 1)
        print(f"Starting Run {run + 1}/{runs}...")

        # 1-based indexing cho tiện giữ nguyên logic hiện tại
        task = [None] * (num_task + 1)
        fitness_arr = [None] * (num_task + 1)
        Nice_gen = np.zeros((generation + 1, num_task + 1))
        r = np.zeros((generation + 1, num_task + 1))
        T = np.zeros((generation + 1, num_task + 1))
        ES = np.zeros(num_task + 1)
        selected_task = np.arange(num_task + 1, dtype=int)

        # ===== Khởi tạo quần thể và fitness ban đầu =====
        for t in range(1, num_task + 1):
            task[t] = np.random.uniform(0.0, 1.0, (population, dim_max))
            fitness_arr[t] = np.zeros(population)

            for j in range(population):
                fitness_arr[t][j] = calculate_fitness(task[t][j], t)

            # diversity tại generation 0
            _, dim, _ = get_task_info(f"T{t}")
            task_coords = task[t][:, :dim]
            centroid = np.mean(task_coords, axis=0)
            distances = np.linalg.norm(task_coords - centroid, axis=1)
            T[0][t] = np.mean(distances)

        # ===== Tiến hóa =====
        for g in range(1, generation + 1):
            # Lưu best fitness TRƯỚC khi bước sang generation g
            # (đường cong hội tụ trung bình qua các run)
            for t in range(1, num_task + 1):
                avg_fitness[g][t] += np.min(fitness_arr[t])

            if g % 100 == 0:
                print(f"  Generation {g}/{generation} completed.")

            # Đúng paper: chỉ kích hoạt ESE + RAKS khi g mod Δg == 0
            if g % delta_g == 0:
                ES = ESE(task, g, Or, Od, Nice_gen, r, T)
                selected_task = RAKS(task, ES)
            else:
                ES.fill(0)
                selected_task = np.arange(num_task + 1, dtype=int)

            task, fitness_arr, Nice_gen[g] = RAKT(
                g,
                task,
                ES,
                fitness_arr,
                selected_task,
                alpha,
                F,
                CR,
                population,
                dim_max,
            )

            # Cập nhật diversity sau khi hoàn tất generation g
            for t in range(1, num_task + 1):
                _, dim, _ = get_task_info(f"T{t}")
                task_coords = task[t][:, :dim]
                centroid = np.mean(task_coords, axis=0)
                distances = np.linalg.norm(task_coords - centroid, axis=1)
                T[g][t] = np.mean(distances)

        # ===== Lưu best fitness cuối cùng của run hiện tại =====
        for t in range(1, num_task + 1):
            matrix_fitness[run, t - 1] = np.min(fitness_arr[t])

    # Ma trận kết quả cuối: runs x tasks
    np.savetxt(
        f"RADE_Matrix_{ACTIVE_PROBLEM_ID}.txt",
        matrix_fitness,
        fmt="%.6e",
        delimiter="\t",
    )

    # Đường cong hội tụ trung bình qua runs
    avg_fitness_matrix = avg_fitness / runs
    for g in range(1, generation + 1):
        log_to_file(avg_fitness_matrix[g, 1:].tolist())

    # AFV cuối cùng: mean best fitness trên K tasks, rồi trung bình qua runs
    final_mean = np.mean(matrix_fitness)
    print(f"\nKết quả: Mean best fitness = {final_mean:.6e}")
    print(f"Output: RADE_{ACTIVE_PROBLEM_ID}.txt")
    print(f"Output: RADE_Matrix_{ACTIVE_PROBLEM_ID}.txt")


if __name__ == "__main__":
    start_time = time.time()
    RADE()
    elapsed = (time.time() - start_time) / 60
    print(f"Thuật toán chạy trong {elapsed:.2f} phút")