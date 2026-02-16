"""
EVAL.PY - Evaluation script for LLMGE

Evaluates tensor decomposition models (like the seed model) and returns fitness metrics.

Fitness Objectives:
1. Correctness ratio (0.0 to 1.0) - MAXIMIZE
2. Number of multiplications (rank) - MINIMIZE

Compatible with LLMGE structure.
"""

import argparse
import importlib
import os
import sys
import numpy as np
import random
from pathlib import Path as p
from os.path import join as pj


def create_save_dir(save_root):
    """Create incrementing exp directory."""
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)
    
    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    
    return save_dir




# BELOW 2 FUNCTIONS ARE FOR CHECKING TENSOR REPRESENTATION CORRECTNESS.

# ===================== Helper to generate the matmul tensor =====================

def generate_matmul_tensor(n):
    dim = n * n
    T = np.zeros((dim, dim, dim), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[i*n + j, j*n + k, k*n + i] = 1

    return T

# ===================== VERIFICATION VIA *TENSOR DECOMP* CORRECTNESS: =====================

def verify_tensor_decomposition(decomposition, n, m, p_dim):
    """
    EXACT AlphaEvolve tensor verification.
    """
    U, V, W = decomposition

    matmul_tensor = np.zeros((n*m, m*p_dim, p_dim*n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p_dim):
                matmul_tensor[i*m + j, j*p_dim + k, k*n + i] = 1
    # matmul_tensor = generate_matmul_tensor(n)

    constructed_tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
    constructed_tensor = np.rint(constructed_tensor).astype(np.int32)

    print("TRUTH:\n")
    print(matmul_tensor)
    print("ACTUAL:\n")
    print(constructed_tensor)

    return np.array_equal(constructed_tensor, matmul_tensor)






# ===================== HELPER FUNCTION TO CONVERT TENSOR FACTORS TO MULT: =====================

def apply_decomposition_to_multiply(A, B, factor_matrix_1, factor_matrix_2, factor_matrix_3):
    """
    Apply tensor decomposition to multiply matrices A and B.
    
    The decomposition represents the matrix multiplication tensor.
    For <n,n,n> multiplication, the factors are (n²×R) matrices.
    
    Key insight: Each column r in the factors represents one "multiplication":
    - factor_matrix_1[:, r] selects elements from A (flattened)
    - factor_matrix_2[:, r] selects elements from B (flattened)
    - factor_matrix_3[:, r] determines where the product goes in C (flattened)
    
    Args:
        A: Matrix of shape (n, n)
        B: Matrix of shape (n, n)
        factor_matrix_1: Shape (n², R)
        factor_matrix_2: Shape (n², R)
        factor_matrix_3: Shape (n², R)
    
    Returns:
        C: Result matrix of shape (n, n)
    """
    n = A.shape[0]
    R = factor_matrix_1.shape[1]  # Number of rank-1 components (multiplications)
    
    # Flatten input matrices
    A_flat = A.flatten()  # Shape: (n²,)
    B_flat = B.flatten()  # Shape: (n²,)
    C_flat = np.zeros(n * n)  # Shape: (n²,)
    
    # Apply each rank-1 component
    for r in range(R):
        u = factor_matrix_1[:, r]  # Select from A
        v = factor_matrix_2[:, r]  # Select from B
        w = factor_matrix_3[:, r]  # Contribute to C
        
        # One scalar multiplication
        Au = np.dot(u, A_flat)
        Bv = np.dot(v, B_flat)
        scalar = Au * Bv
        
        # Add contribution to result
        C_flat += scalar * w
    
    # Reshape back to matrix
    # print("C_flat:", C_flat)
    # print("C_flat reshaped row-major:\n", C_flat.reshape(n, n))
    print("C_flat reshaped col-major:\n", C_flat.reshape(n, n, order="F"))

    C = C_flat.reshape(n, n, order="F")  #ROW MAJOR ORDER!!!!!
    return C

# ===================== VERIFICATION VIA RANDOM MULTIPLICATION TESTS: =====================
def verify_decomposition(factor_matrix_1, factor_matrix_2, factor_matrix_3, 
                         n=2, num_tests=50, tol=5e-6):
    """
    Verify that the decomposition correctly multiplies matrices.
    
    Tests on random matrices and returns detailed statistics.
    
    Args:
        factor_matrix_1, factor_matrix_2, factor_matrix_3: Decomposition factors
        n: Matrix dimension
        num_tests: Number of random tests
        tol: Error tolerance
    
    Returns:
        correct_ratio: Fraction of tests passed (0.0 to 1.0)
        num_correct: Number of tests passed
        num_total: Total tests
        max_error: Maximum error observed
        avg_error: Average error
        num_multiplications: Rank of decomposition
    """
    num_correct = 0
    max_error = 0.0
    total_error = 0.0
    
    # Count multiplications (rank of decomposition)
    num_multiplications = factor_matrix_1.shape[1]
    
    for _ in range(num_tests):
        # Generate random test matrices
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Ground truth
        C_expected = A @ B

        print("\nGROUND TRUTH MULT")
        print(C_expected)
        
        print("\nACTUAL PRODUCT:")
        # Use decomposition
        C_result = apply_decomposition_to_multiply(
            A, B, factor_matrix_1, factor_matrix_2, factor_matrix_3
        )
        
        # print(C_result)
        # Compute error
        error = np.max(np.abs(C_result - C_expected))
        max_error = max(max_error, error)
        total_error += error
        
        if error <= tol:
            num_correct += 1
    
    correct_ratio = num_correct / num_tests
    avg_error = total_error / num_tests
    
    return correct_ratio, num_correct, num_tests, max_error, avg_error, num_multiplications


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="seedModel1", help="model file")
    parser.add_argument('--save_dir', type=str, default="trained", help="path where results will be saved")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    parser.add_argument('--variant_dir', type=str, default='models', help="directory where models are written by LLM-GE")
    
    # Problem parameters
    parser.add_argument('--N', type=int, default=2, help="matrix dimension")
    parser.add_argument('--R', type=int, default=7, help="target rank")
    parser.add_argument('--num_tests', type=int, default=50, help="number of verification tests")
    
    return parser.parse_args()


if __name__ == '__main__':
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)
    args = get_args()
    
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Import model dynamically
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)
    
    # Get gene_id
    try:
        gene_id = args.model.split('model_')[1]
    except:
        gene_id = 'seed'
    
    save_dir = f'{args.save_dir}/{gene_id}'
    create_save_dir(save_dir)
    
    run_dir = p(args.save_dir) / f"{gene_id}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)





    print("="*120)
    print(f"EVALUATING: {args.N}×{args.N} matrix multiplication")
    print(f"Target rank: {args.R}")
    print(f"Gene ID: {gene_id}")
    print("="*120)
    
    # ========================================================================
    # RUN THE MODEL
    # ========================================================================
    
    # The model should have a main() function or similar that returns factors
    # We'll call it and capture the results
    print("\nRunning model optimization...")
    
    # Temporarily redirect to capture model output if needed
    import io
    import contextlib
    
    # Save original stdout
    original_stdout = sys.stdout
    
    # Run the model's main function
    # The seed model has a main() function that we can call
    try:
        # Call the model's main function
        # It will create its own experiment directory and save results
        # model_module.main()
        old_argv = sys.argv
        sys.argv = [
            old_argv[0],
            "--save_dir", str(run_dir),
            "--N", str(args.N),
            "--R", str(args.R),
        ]
        model_module.main()
        sys.argv = old_argv







        
        # The model saves results in its own exp directory
        # We need to find the most recent one
        # ===========================
        # trained_dir = p("trained")
        # exp_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])
        
        # if not exp_dirs:
        #     raise FileNotFoundError("No experiment directories found")
        
        # latest_exp_dir = exp_dirs[-1]
        
        # # Load the saved factors
        # factors_file = latest_exp_dir / "factors.npz"
        
        # if not factors_file.exists():
        #     raise FileNotFoundError(f"No factors.npz found in {latest_exp_dir}")

        # instead: -------------
        factors_file = run_dir / "factors.npz"
        if not factors_file.exists():
            raise FileNotFoundError(f"No factors found in {run_dir}")

        # ===========================
        
        factors_data = np.load(factors_file)
        factor_matrix_1 = factors_data['U']
        factor_matrix_2 = factors_data['V']
        factor_matrix_3 = factors_data['W']
        
        # REMOVING ROUNDING FIXED MY ERRORS!
        # def round_half(x):
        #     return np.round(x * 2) / 2
        
        # factor_matrix_1 = round_half(factor_matrix_1)
        # factor_matrix_2 = round_half(factor_matrix_2)
        # factor_matrix_3 = round_half(factor_matrix_3)
        
        print(f"\nLoaded factors from {factors_file}")
        print(f"Factor shapes: {factor_matrix_1.shape}, {factor_matrix_2.shape}, {factor_matrix_3.shape}")
        
    except Exception as e:
        print(f"\nERROR loading factors: {e}")
        print("Using dummy factors for demonstration...")
        
        # Create dummy factors (standard algorithm)
        dim = args.N * args.N
        rank = args.R
        factor_matrix_1 = np.random.randn(dim, rank)
        factor_matrix_2 = np.random.randn(dim, rank)
        factor_matrix_3 = np.random.randn(dim, rank)
    
    # ========================================================================
    # VERIFY CORRECTNESS
    # ========================================================================
    
    print(f"\nVerifying correctness using seed model's verification method...")
    
    # ============= USE THIS FOR VERIFICATION VIA TENSOR! ============
    # is_correct = verify_tensor_decomposition(
    #     decomposition=(factor_matrix_1, factor_matrix_2, factor_matrix_3),
    #     n=args.N,
    #     m=args.N,
    #     p_dim=args.N
    # )
    # num_multiplications = args.R


    # ============= USE THIS FOR VERIFICATION VIA RANDOM MATRIX MULT CHECKS! ============
    correct_ratio, num_correct, num_total, max_error, avg_error, num_multiplications = verify_decomposition(
        factor_matrix_1, factor_matrix_2, factor_matrix_3,
        n=args.N,
        num_tests=args.num_tests
    )    
    # print("PERCENTAGE CORRECT!!!!\n")
    # print(correct_ratio)
    standard_mults = args.N ** 3  # Standard algorithm for n×n matrices
    
    # ========================================================================
    # CALCULATE FITNESS
    # ========================================================================
    
    # Fitness has two objectives:
    # 1. Maximize correctness_ratio (0.0 to 1.0)
    # 2. Minimize num_multiplications
    
    # We'll use a weighted combination:
    # - Correctness is most important (huge penalty if wrong)
    # - Number of multiplications matters once correct
    
    if correct_ratio == 1.0:
        # Perfect: optimize for performance
        # Scale num_multiplications to be comparable to small differences in ratio
        fitness = num_multiplications * 1000
        status = "FULLY CORRECT"
        
    elif correct_ratio >= 0.8:
        # Mostly correct: medium penalty
        penalty = (1.0 - correct_ratio) * 100000
        fitness = penalty + num_multiplications * 1000 + avg_error * 10000
        status = f"MOSTLY CORRECT ({correct_ratio:.0%})"
        
    else:
        # Mostly wrong: large penalty
        penalty = (1.0 - correct_ratio) * 1000000
        fitness = penalty + avg_error * 10000 + num_multiplications
        status = f"INCORRECT ({correct_ratio:.0%})"
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    print("\n" + "="*120)
    print("RESULTS")
    print("="*120)
    print(f"Rank (multiplications): {num_multiplications}")
    print(f"Target rank: {args.R}")
    print(f"Standard algorithm: {standard_mults} multiplications")
    print(f"Improvement: {100*(1 - num_multiplications/standard_mults):.1f}% reduction")
    
    # ===========================
    # print(f"\nCorrectness (using seed model's verification):")
    # if is_correct:
    #    fitness = num_multiplications * 1000  # e.g., 7000 for rank-7
    # else:
    #    fitness = 1000000 + num_multiplications  # e.g., 1000007 if wrong
    # ===========================
    print(f"\nCorrectness (via random mult tests):")
    print(f"Passed {num_correct}/{num_total} tests.")
    # ============================
    
    print(f"\nFitness for LLMGE (MINIMIZE):")
    # print(f"  {status}")
    print(f"  Fitness score: {fitness:.4f}")
    
    if correct_ratio == 1.0:
        print(f"  Breakdown: {num_multiplications} mults × 1000")
    elif correct_ratio >= 0.8:
        penalty = (1.0 - correct_ratio) * 100000
        print(f"  Breakdown: {penalty:.2f} (penalty) + {num_multiplications * 1000} (mults) + {avg_error * 10000:.2f} (error)")
    else:
        penalty = (1.0 - correct_ratio) * 1000000
        print(f"  Breakdown: {penalty:.2f} (penalty) + {avg_error * 10000:.2f} (error) + {num_multiplications} (mults)")
    
    print("="*120)
    
    # Print fitness for LLMGE to parse
    print(f"\n{fitness:.4f}")
    # print(f"# Breakdown: ratio={correct_ratio:.4f}, rank={num_multiplications}, error={avg_error:.2e}")
    
    # Save results
    results_text = f"{fitness:.4f}"
    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    dir_path = os.path.dirname(filename)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(filename, 'w') as file:
        file.write(results_text)
    
    print(f"\nResults written to {filename}")
    print('='*120)
    print('job done')
    print('='*120)