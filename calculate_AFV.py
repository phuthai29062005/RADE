import argparse
from pathlib import Path
import numpy as np


def load_matrix(file_path: str | Path, expected_tasks: int = 50) -> np.ndarray:
    """
    Load a result matrix and normalize it to shape: (runs, tasks).

    Accepted layouts:
    - runs x tasks  (preferred)
    - tasks x runs  (will be transposed)
    - flat vector   (will be reshaped if divisible by expected_tasks)
    """
    path = Path(file_path)
    data = np.loadtxt(path, dtype=float)

    if data.ndim == 1:
        if data.size % expected_tasks != 0:
            raise ValueError(
                f"{path.name}: cannot reshape {data.size} values into (?, {expected_tasks})."
            )
        data = data.reshape(-1, expected_tasks)

    rows, cols = data.shape

    if cols == expected_tasks:
        matrix = data  # already runs x tasks
    elif rows == expected_tasks:
        matrix = data.T  # tasks x runs -> runs x tasks
    else:
        flat = data.reshape(-1)
        if flat.size % expected_tasks != 0:
            raise ValueError(
                f"{path.name}: shape {data.shape} is incompatible with expected_tasks={expected_tasks}."
            )
        matrix = flat.reshape(-1, expected_tasks)

    return matrix


def compute_afv(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    matrix shape: (runs, tasks)
    AFV per run = mean of final best fitness across tasks.
    Overall AFV = mean of AFV over runs.
    """
    afv_per_run = np.mean(matrix, axis=1)
    afv_mean = float(np.mean(afv_per_run))
    afv_std = float(np.std(afv_per_run, ddof=1)) if len(afv_per_run) > 1 else 0.0
    return afv_per_run, afv_mean, afv_std


def summarize_problem(file_path: str | Path, expected_tasks: int = 50) -> dict:
    matrix = load_matrix(file_path, expected_tasks=expected_tasks)
    afv_per_run, afv_mean, afv_std = compute_afv(matrix)
    return {
        "file": Path(file_path).name,
        "runs": int(matrix.shape[0]),
        "tasks": int(matrix.shape[1]),
        "afv_per_run": afv_per_run,
        "afv_mean": afv_mean,
        "afv_std": afv_std,
    }


def write_table(results: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    lines = []
    header = f"{'Problem':<12} | {'Runs':>4} | {'Tasks':>5} | {'AFV Mean':>14} | {'AFV Std':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for idx, result in enumerate(results, start=1):
        label = f"CEC19-P{idx}"
        lines.append(
            f"{label:<12} | {result['runs']:>4} | {result['tasks']:>5} | "
            f"{result['afv_mean']:>14.6e} | {result['afv_std']:>14.6e}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AFV from RADE_Matrix files."
    )
    parser.add_argument(
        "--pattern",
        default="RADE_Matrix_{}.txt",
        help="Filename pattern for problem matrices. Use {} as the problem index placeholder.",
    )
    parser.add_argument(
        "--problems",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Problem indices to process.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=50,
        help="Expected number of tasks per problem.",
    )
    parser.add_argument(
        "--output",
        default="RADE_Table_Results.txt",
        help="Output summary file.",
    )
    args = parser.parse_args()

    results = []

    for problem_id in args.problems:
        file_path = args.pattern.format(problem_id)
        path = Path(file_path)
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue

        result = summarize_problem(path, expected_tasks=args.tasks)
        results.append(result)

        print(
            f"CEC19-P{problem_id}: runs={result['runs']}, tasks={result['tasks']}, "
            f"AFV={result['afv_mean']:.6e}, std={result['afv_std']:.6e}"
        )

    if not results:
        raise FileNotFoundError("No valid matrix files were found.")

    write_table(results, args.output)
    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
