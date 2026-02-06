"""
EVAL_JAX.PY

Full evaluation script for AlphaEvolve / LLMGE-style tensor decomposition search.

Responsibilities:
- Optimize low-rank tensor decompositions with JAX autodiff
- Verify exactness of multiplication
- Count scalar multiplications (rank R)
- Compare to standard and Strassen multiplication
- Save only exact decompositions
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import json
import os
import argparse


# ---------------------------
# Helper functions
# ---------------------------

def create_multiplication_tensor(m, n, p):
    """
    Construct the 3D tensor representing matrix multiplication:
    T[a_idx, b_idx, c_idx] = 1 if c = A @ B else 0
    """
    T = np.zeros((m * n, n * p, m * p), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                a_idx = i * n + j
                b_idx = j * p + k
                c_idx = i * p + k
                T[a_idx, b_idx, c_idx] = 1.0
    return jnp.array(T)


def verify_algorithm(m, n, p, decomposition, tests=50, tol=1e-6):
    """
    Verify that the decomposition exactly multiplies matrices.
    Returns: correct (bool), max_error (float)
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

        C_calc = C_flat.reshape(m, p)
        err = np.max(np.abs(C_calc - C_true))
        max_error = max(max_error, err)
        if err > tol:
            return False, max_error

    return True, max_error


def round_decomposition(decomposition):
    """
    Round decomposition to half-integers (as AlphaEvolve does).
    """
    def round_half(x):
        return np.round(x * 2) / 2
    rounded = [(round_half(np.array(u)), round_half(np.array(v)), round_half(np.array(w)))
               for u, v, w in decomposition]
    return rounded


def save_algorithm(decomposition, path):
    """
    Save exact decomposition as JSON.
    """
    serializable = [
        {"u": u.tolist(), "v": v.tolist(), "w": w.tolist()}
        for u, v, w in decomposition
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


# ---------------------------
# Standard & Strassen multiplication helpers
# ---------------------------

def standard_mult(A, B):
    """Standard matrix multiplication."""
    C = A @ B
    num_mults = A.shape[0] * A.shape[1] * B.shape[1]
    return C, num_mults


def strassen_2x2(A, B):
    """Strassen algorithm for 2x2 matrices."""
    assert A.shape == (2, 2) and B.shape == (2, 2)
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    e, f, g, h = B[0, 0], B[0, 1], B[1, 0], B[1, 1]

    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (f - h)
    M4 = d * (g - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)

    C = np.array([
        [M1 + M4 - M5 + M7, M3 + M5],
        [M2 + M4, M1 - M2 + M3 + M6]
    ])
    num_mults = 7
    return C, num_mults


# ---------------------------
# JAX optimizer
# ---------------------------

def optimize_jax(m, n, p, rank, num_steps=2000, lr=1e-2, seed=0):
    """
    Optimize a rank-R decomposition for matrix multiplication using JAX autodiff.
    Returns: factors = [(u,v,w), ...], final loss
    """
    key = jax.random.PRNGKey(seed)
    target_tensor = create_multiplication_tensor(m, n, p)

    # Initialize rank-R factors
    factors = [(jax.random.normal(key, (m*n,)),
                jax.random.normal(key, (n*p,)),
                jax.random.normal(key, (m*p,))) for _ in range(rank)]

    # Define loss function
    def loss_fn(factors):
        loss = 0.0
        for u, v, w in factors:
            reconstructed = jnp.einsum('i,j,k->ijk', u, v, w)
            loss += jnp.mean((reconstructed - target_tensor) ** 2)
        return loss

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Optimization loop
    for step in range(num_steps):
        loss_val, grads = loss_and_grad(factors)
        factors = [(u - lr * du, v - lr * dv, w - lr * dw)
                   for (u, v, w), (du, dv, dw) in zip(factors, grads)]
        if step % 500 == 0:
            print(f"Step {step:4d} | loss = {loss_val:.6f}")

    # Convert to numpy
    factors_np = [(np.array(u), np.array(v), np.array(w)) for u, v, w in factors]
    return factors_np, float(loss_val)


# ---------------------------
# Main evaluation
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Optimize decomposition
    start_time = time.time()
    factors, final_loss = optimize_jax(args.m, args.n, args.p, args.rank,
                                       num_steps=args.num_steps, lr=args.lr, seed=args.seed)
    runtime = time.time() - start_time

    # Round factors for exact algorithm
    factors_rounded = round_decomposition(factors)

    # Verify exactness
    correct, max_error = verify_algorithm(args.m, args.n, args.p, factors_rounded)

    # Count multiplications (rank)
    num_mults = len(factors_rounded)

    print("\nRESULTS")
    print(f"Runtime: {runtime:.4f} s")
    print(f"Rank (scalar multiplications): {num_mults}")
    print(f"Correct: {correct}")
    print(f"Max error observed: {max_error:.2e}")

    # Compare to standard multiplication
    A_test = np.random.randn(args.m, args.n)
    B_test = np.random.randn(args.n, args.p)
    _, std_mults = standard_mult(A_test, B_test)
    print(f"Standard algorithm multiplications: {std_mults}")

    if args.m == 2 and args.n == 2 and args.p == 2:
        _, strassen_mults = strassen_2x2(A_test, B_test)
        print(f"Strassen 2x2 multiplications: {strassen_mults}")

    # Save exact decomposition if correct
    if correct:
        filename = f"successful_algorithms/mm_{args.m}x{args.n}x{args.p}_r{num_mults}.json"
        save_algorithm(factors_rounded, filename)
        print("Exact decomposition saved:", filename)
    else:
        print("Decomposition is not exact; not saved.")


if __name__ == "__main__":
    main()
