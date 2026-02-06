"""
EVAL.PY

Responsible for:
- Running training via LLMGE
- Verifying exact correctness
- Counting scalar multiplications
- Saving successful algorithms
- Printing metrics (correctness, multiplications, runtime)
"""

import numpy as np
import importlib
import argparse
import os
import json
import time


def verify_algorithm(m, n, p, decomposition, tests=50, tol=1e-6):
    """
    Check whether the decomposition exactly implements matrix multiplication.
    Returns (is_correct, max_error)
    """
    max_error = 0.0
    for _ in range(tests):
        A = np.random.randn(m, n)
        B = np.random.randn(n, p)
        C_true = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()
        C_flat = np.zeros(m * p)

        for u, v, w in decomposition:
            C_flat += (u @ A_flat) * (v @ B_flat) * w

        error = np.max(np.abs(C_flat.reshape(m, p) - C_true))
        max_error = max(max_error, error)
        if error > tol:
            return False, max_error

    return True, max_error


def count_multiplications(decomposition):
    """
    The number of scalar multiplications is exactly the rank (R)
    """
    return len(decomposition)


def save_algorithm(decomposition, path):
    """
    Save a successful exact algorithm to disk as JSON
    """
    serializable = [
        {"u": u.tolist(), "v": v.tolist(), "w": w.tolist()}
        for u, v, w in decomposition
    ]
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="AE_model")
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tests", type=int, default=50)
    args = parser.parse_args()

    model_module = importlib.import_module(args.model)
    model = model_module.Model(args.m, args.n, args.p, args.rank)

    print(f"\nTraining {args.m}x{args.n} x {args.n}x{args.p} decomposition (rank={args.rank})")
    start_time = time.perf_counter()
    model.fit(num_steps=2000, lr=1e-2, seed=args.seed)
    end_time = time.perf_counter()

    decomposition = model.get_decomposition()
    R = count_multiplications(decomposition)

    # Verify correctness
    correct, max_error = verify_algorithm(args.m, args.n, args.p, decomposition, tests=args.num_tests)

    # Print metrics
    print("\n=== EVALUATION METRICS ===")
    print(f"Correct: {correct}")
    print(f"Rank / scalar multiplications: {R}")
    print(f"Maximum observed error: {max_error:.2e}")
    print(f"Runtime: {end_time - start_time:.4f} sec")
    print("==========================\n")

    # Save if exact
    if correct:
        os.makedirs("successful_algorithms", exist_ok=True)
        filename = f"successful_algorithms/mm_{args.m}x{args.n}x{args.p}_r{args.rank}.json"
        save_algorithm(decomposition, filename)
        print("Exact algorithm found and saved:", filename)
    else:
        print("Algorithm not exact.")


if __name__ == "__main__":
    main()
