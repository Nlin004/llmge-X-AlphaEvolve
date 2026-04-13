"""
EVAL.PY - Evaluation script for LLMGE

Evaluates tensor decomposition models (like the seed model) and returns fitness metrics.

Fitness Objectives:
1. Elementwise matrix equality ratio (0.0 to 1.0) - MINIMIZE
2. RMSE of reconstructed tensor vs true tensor - MINIMIZE
3. Number of basis multiplication tests passed - MAXIMIZE
4. Number of random multiplication tests passed - MAXIMIZE

Compatible with LLMGE structure, if cfg configured for a 4-tuple fitness value
"""

import argparse
import importlib
import os
import sys
import time
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


def project_to_integer_factors(U, V, W):
    U_int = np.rint(U).astype(np.int32)
    V_int = np.rint(V).astype(np.int32)
    W_int = np.rint(W).astype(np.int32)
    return U_int, V_int, W_int

def project_to_half_integer_factors(U, V, W):
    U_half = np.round(U * 2) / 2
    V_half = np.round(V * 2) / 2
    W_half = np.round(W * 2) / 2
    return U_half, V_half, W_half


# RECTANGULAR SUPPORT:
def exact_tensor_check(U, V, W, n, m, p):  # Added m, p parameters!
    """Check tensor for N×M × M×P multiplication."""
    dim_A = n * m
    dim_B = m * p
    dim_C = n * p
    
    T_true = np.zeros((dim_A, dim_B, dim_C), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, i*p+k] = 1
    
    # This is CP decomposition! 
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    # T_hat = np.rint(T_hat)  # [1, 0, ...]  # do NOT round the final tensor!!!!

    print("\nTarget tensor (T_true):")
    print(T_true)
    print("\nReconstructed tensor (T_hat):")
    print(T_hat)

    # Count incorrect entries
    total_entries = T_true.size
    incorrect = np.sum(~np.isclose(T_hat, T_true, atol=1e-10))
    fraction_incorrect = incorrect / total_entries

    print(f"\nfrac: {incorrect}/{total_entries} wrong")

    # Calculate RMSE as tie-breaker - "magnitude" of the errors in the tensor. 
    # All wrong but within magnitude of like 0.5 should still be more valid than same wrong but avg deviation of like 1-5.
    rmse = np.sqrt(np.mean((T_hat - T_true) ** 2))

    return fraction_incorrect, rmse
    # return np.array_equal(T_hat, T_true)


# Is the tensor from (U,V,W) exactly equal to the tensor from (U_ref,V_ref,W_ref) in this basis?
def exact_tensor_check_against_ref(U, V, W, T_ref):
    """
    Check if candidate (U,V,W) reproduces the same tensor as T_ref.
    """
    print("Reference tensor (T_ref):")
    print(T_ref)
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    # If everything is integral/half-integral, this can be exact equality.
    print("Reconstructed tensor (T_hat):")
    print(T_hat)
    return np.array_equal(T_hat, T_ref)
    # define “correctness” as “equivalent to the reference algorithm,” not necessarily the canonical A @ B with the chosen flattening.

    # doing the T_ref to check -> does the candidate reconstruct the same tensor as this possibly wrong Strassen variant

def apply_decomposition_bilinear(A, B, U, V, W):
    # this actually applies the decomposition of U V W to multiply A and B, by treating U V W as defining a bilinear algorithm for matrix multiplication. 
    # This kinda just tests the actual multiplication aspect - not just checking if the tensors match but checking if the actual multiplication results match for all basis pairs and random pairs.
    # n = A.shape[0]

    n, m1 = A.shape  # A is n×m
    m2, p = B.shape  # B is m×p

    assert m1 == m2, f"Incompatible dimensions: A is {n}×{m1}, B is {m2}×{p} and {m1}!={m2}."
    m = m1
    
    # ROW-MAJOR (default) - matches your Strassen factors
    A_flat = A.flatten()  # NO order='F'!
    B_flat = B.flatten()  # NO order='F'!
    # C_flat = np.zeros(n * n)
    C_flat = np.zeros(n * p) #recntagular support
    
    R = U.shape[1]
    
    for r in range(R):
        a_r = np.dot(U[:, r], A_flat)
        b_r = np.dot(V[:, r], B_flat)
        C_flat += (a_r * b_r) * W[:, r] # ONE MULTIPLCATION DONE! this loops R times, so we guarantee r mults.
    
    # return C_flat.reshape(n, n)  # NO order='F'!
    return C_flat.reshape(n, p)  # RECNTAGULAR SUPORTO


# ============ BASIS COMPARISON TESTS =============

def basis_verification_cleaned(U, V, W, n, m, p):
    print("**** BASIS TESTS **** ")
    # total_tests = n * n * n * n
    total_tests = n * m * m * p
    counter = 0

    for i in range(n):
        for j in range(m):
            A = np.zeros((n, m), dtype=np.int64)
            A[i, j] = 1

            for k in range(m):
                for l in range(p):
                    B = np.zeros((m, p), dtype=np.int64)
                    B[k, l] = 1

                    C_expected = A @ B
                    C_actual = apply_decomposition_bilinear(A, B, U, V, W).astype(np.int64)

                    # print("\nC_expected:\n", C_expected)
                    # print("C_actual:\n", C_actual)
                    # print("-"*20)

                    if np.array_equal(C_actual, C_expected):
                        counter += 1
    print(f"Basis verification score: {counter}/{total_tests} basis tests correct.\n\n")
    return counter/total_tests


def apply_reference(A, B, U_ref, V_ref, W_ref):
    return apply_decomposition_bilinear(A, B, U_ref, V_ref, W_ref)
def basis_verification_vs_ref(U, V, W, U_ref, V_ref, W_ref, n):
    total = n*n*n*n
    counter = 0
    for i in range(n):
        for j in range(n):
            A = np.zeros((n,n), dtype=int)
            A[i,j] = 1
            for k in range(n):
                for l in range(n):
                    B = np.zeros((n,n), dtype=int)
                    B[k,l] = 1
                    C_expected = apply_reference(A,B,U_ref,V_ref,W_ref)
                    C_actual   = apply_decomposition_bilinear(A,B,U,V,W)
                    if np.array_equal(C_actual, C_expected):
                        counter += 1
    return counter / total



# ============ RANDOM MATRIX TESTS =============
def random_matrix_verification_vs_ref(U, V, W, U_ref, V_ref, W_ref,
                                      n, num_tests=50, seed=0):
    rng = np.random.default_rng(seed)
    counter = 0
    for _ in range(num_tests):
        A = rng.integers(-3, 4, size=(n,n), dtype=int)
        B = rng.integers(-3, 4, size=(n,n), dtype=int)
        C_expected = apply_reference(A,B,U_ref,V_ref,W_ref)
        C_actual   = apply_decomposition_bilinear(A,B,U,V,W)
        if np.array_equal(C_actual, C_expected):
            counter += 1
    return counter / num_tests

# def random_matrix_verification(U, V, W, n, m, p, num_tests=50, seed=0):
#     rng = np.random.default_rng(seed)
#     print("**** RANDOM MATRIX TESTS ****")
#     counter = 0
#     for _ in range(num_tests):
#         A = rng.integers(-3, 4, size=(n, m), dtype=np.int64)
#         B = rng.integers(-3, 4, size=(m, p), dtype=np.int64)

#         C_expected = A @ B
#         C_actual = apply_decomposition_bilinear(A, B, U, V, W)

#         # print("C_expected:\n", C_expected)
#         # print("C_actual:\n", C_actual)
#         # print("-"*20)

#         if np.array_equal(C_actual, C_expected):
#             counter += 1
#     print(f"Random matrix verification score: {counter}/{num_tests} tests correct.")
#     return counter/num_tests # float

# random matrix verification with timing and median for comparisons to other correct algos.
def random_matrix_verification(U, V, W, n, m, p, num_tests=50, seed=0, repeats=200):
    rng = np.random.default_rng(seed)
    print("**** RANDOM MATRIX TESTS ****")
    counter = 0
    times = []

    for _ in range(num_tests):
        A = rng.integers(-3, 4, size=(n, m), dtype=np.int64)
        B = rng.integers(-3, 4, size=(m, p), dtype=np.int64)
        C_expected = A @ B

        # Repeat each multiplication to average out Python/OS noise
        t0 = time.perf_counter_ns()
        for _ in range(repeats):
            C_actual = apply_decomposition_bilinear(A, B, U, V, W)
        elapsed_ns = (time.perf_counter_ns() - t0) / repeats

        times.append(elapsed_ns)

        if np.array_equal(C_actual, C_expected):
            counter += 1

    median_ns = int(np.median(times))
    print(f"Random matrix verification score: {counter}/{num_tests} tests correct.")
    print(f"Median wall-clock time per multiply: {median_ns} ns")
    return counter / num_tests, median_ns


# metric: count the number of total additions.
def count_operations(U, V, W):
    rank = U.shape[1]
    adds = sum(np.count_nonzero(U[:, k]) - 1 for k in range(rank))
    adds += sum(np.count_nonzero(V[:, k]) - 1 for k in range(rank))
    adds += sum(np.count_nonzero(W[:, k]) - 1 for k in range(rank))
    return adds



# ============ MAIN VERIFICATION PIPELINE =============

def verify_decomposition_pipeline(U_float, V_float, W_float, n, m, p, num_random_tests=50):
    # Step 0: projection / rounding first
    # U, V, W = project_to_integer_factors(U_float, V_float, W_float)

    print("First rounding to half-integers.\n")

    U_half, V_half, W_half = U_float, V_float, W_float # testing temporarily with no rounding.
    USE_ROUNDING = True # rounding FACTORS is allowed. try to match with half integer tensors.
    if USE_ROUNDING:
        U_half, V_half, W_half = project_to_half_integer_factors(U_float, V_float, W_float)

    print("U_half:\n", U_half)
    print("V_half:\n", V_half)
    print("W_half:\n", W_half)
    # T_ref = compute_reference_tensor(U_float, V_float, W_float)


    # Step 1: actual tensor correctness
    ratio_matrix_entries_wrong, magnitude_of_errors = exact_tensor_check(U_half, V_half, W_half, n, m, p)
    print(f"Exact tensor correctness check: {100*(1-ratio_matrix_entries_wrong):.2f}% entries correct, RMSE: {magnitude_of_errors:.2e}\n\n")
    # ================ if just checking EQUALS or NOT: ================
    # tensor_check = 0
    # if exact_tensor_check(U_half, V_half, W_half, n):  # if using: change the code in the function to just return np.equals() instead of the ratio + magnitude.
    ## if exact_tensor_check_against_ref(U_half, V_half, W_half, T_ref):
    #     print("[PASS] Exact tensor MATCHES!")
    #     tensor_check = 1
    # else:
    #     print("[FAIL] Exact tensor does NOT match.")
    # ================================================================


    # Step 2: basis verification
    basis_score = basis_verification_cleaned(U_half, V_half, W_half, n, m, p)
    # basis_score = basis_verification_vs_ref(U_half, V_half, W_half, U_ref, V_ref, W_ref, n) # using T_ref instead of canonical A@B as the basis for correctness, since the model is really trying to match T_ref not necessarily the canonical tensor.

    # Step 3: random tests w/ runtime measurement
    # random_score = random_matrix_verification(U_half, V_half, W_half, n, m, p, num_random_tests)
    random_score, median_ns = random_matrix_verification(U_half, V_half, W_half, n, m, p, num_random_tests)

    # random_score = random_matrix_verification_vs_ref(U_half, V_half, W_half, U_ref, V_ref, W_ref, n, num_random_tests) # using T_ref instead of canonical A@B as the basis for correctness, since the model is really trying to match T_ref not necessarily the canonical tensor.
    
    # Step 4: analytical operation count (ADDITIONS!)
    additions = count_operations(U_half, V_half, W_half) # minimize


    # Done: return all metrics gathered from the suites of tests as a 6-tuple for LLMGE 
    fitness = (ratio_matrix_entries_wrong, magnitude_of_errors, basis_score, random_score, median_ns, additions)
    return fitness



def compute_reference_tensor(U_ref, V_ref, W_ref):
    """
    Given reference factors (U_ref, V_ref, W_ref) of shape (n^2, R),
    compute and return the reference tensor T_ref with shape (n^2, n^2, n^2).
    """
    T_ref = np.einsum('ir,jr,kr->ijk', U_ref, V_ref, W_ref)
    return T_ref

# Debugging functions - solve for correct W given U V and the matrices A B
# ==============================================================================
def compute_Ms(U, V, A, B):
    A_flat = A.flatten().astype(float)
    B_flat = B.flatten().astype(float)
    R = U.shape[1]
    Ms = np.zeros(R, float)
    for r in range(R):
        a_r = U[:,r] @ A_flat
        b_r = V[:,r] @ B_flat
        Ms[r] = a_r * b_r
    return Ms

def solve_W_row_for_entry(U, V, entry_idx):
    # entry_idx: 0 for c11, 1 for c12, 2 for c21, 3 for c22
    n = 2
    eqs = []
    rhs = []
    for i in range(n):
        for j in range(n):
            A = np.zeros((n,n), int)
            A[i,j] = 1
            for k in range(n):
                for l in range(n):
                    B = np.zeros((n,n), int)
                    B[k,l] = 1
                    Ms = compute_Ms(U, V, A, B)
                    C = A @ B
                    eqs.append(Ms)
                    rhs.append(C.flatten()[entry_idx])
    eqs = np.stack(eqs, axis=0)    # shape (16,7)
    rhs = np.array(rhs, float)     # shape (16,)

    # Solve least-squares; for exact integer solution you can round
    x, *_ = np.linalg.lstsq(eqs, rhs, rcond=None)
    return x
# ========================================================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="seedModel1", help="model file")
    parser.add_argument('--save_dir', type=str, default="trained", help="path where results (the tensor factors) will be saved after running the model")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    parser.add_argument('--variant_dir', type=str, default='models', help="directory where models are written by LLM-GE")
    
    # Problem parameters - don't define them here, define them ONLY in seedModel!!!
    # parser.add_argument('--N', type=int, default=3, help="matrix dimension")
    # parser.add_argument('--R', type=int, default=23, help="target rank")
    # parser.add_argument('--num_tests', type=int, default=50, help="number of random verification tests")
    
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
    
    # run_dir = p(args.save_dir) / f"{gene_id}_pid{os.getpid()}"
    run_dir = p(args.save_dir).resolve() / f"{gene_id}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)





    print("="*120)
    # print(f"EVALUATING: {args.N}×{args.N} matrix multiplication")
    # print(f"Target rank: {args.R}")
    print(f"Gene ID: {gene_id}")
    print("="*120)
    
    # ========================================================================
    # RUN THE MODEL
    # ========================================================================
    
    # The model should have a main() function or similar that returns factors
    # We'll call it and capture the results
    print("\nRunning model for eval...")
    
    # Temporarily redirect to capture model output if needed
    import io
    import contextlib
    
    # Save original stdout
    original_stdout = sys.stdout
    
    # Run the model's main function
    # The seed model has a main() function that we can call

    current_dir = os.getcwd()
    print(f"Current directory before calling model.main(): {current_dir}")
    try:
        # Call the model's main function
        # It will create its own experiment directory and save results
        old_argv = sys.argv

        sys.argv = [
            old_argv[0],
            "--save_dir", str(run_dir),
        ]
        # Make sure we're in the right directory
        os.chdir(script_directory)
        model_module.main()

        # Find factors file, load it.
        factors_file = run_dir / "factors.npz"
        if not factors_file.exists():
            raise FileNotFoundError(f"No factors found in {run_dir}")
        
        factors_data = np.load(factors_file)
        print(f"\nLoaded factors from {factors_file}")

    except Exception as e: # if can't find factors for some reason, or some other error.
        print(f"\nERROR loading factors: {e}")
        print(f"Looked for factors at: {factors_file}")
        print(f"run_dir contents:")
        if run_dir.exists():
            print(list(run_dir.iterdir()))
        else:
            print(f"  run_dir does not exist: {run_dir}")
        
        # Also check if model saved to default location
        print(f"\nChecking default 'trained' directory:")
        trained_dir = p("trained")
        if trained_dir.exists():
            exp_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir()])
            print(f"  Found directories: {exp_dirs}")
            if exp_dirs:
                latest = exp_dirs[-1]
                print(f"  Latest: {latest}")
                if (latest / "factors.npz").exists():
                    print(f"  [GOOD] factors.npz exists in {latest}")
        
    # ========================================================================
    # VERIFY CORRECTNESS
    # ========================================================================
    factor_matrix_1 = factors_data['U']
    factor_matrix_2 = factors_data['V']
    factor_matrix_3 = factors_data['W']
    # thisN = int(factors_data['N'])
    # thisTargetRank = int(factors_data["R"])
    thisN = int(factors_data['N'])
    thisM = int(factors_data['M'])
    thisP = int(factors_data['P'])
    thisR = int(factors_data['R'])

    USE_ROUNDED = False
    def round_half(x):
        return np.round(x * 2) / 2
    
    if USE_ROUNDED:
        factor_matrix_1 = round_half(factor_matrix_1)
        factor_matrix_2 = round_half(factor_matrix_2)
        factor_matrix_3 = round_half(factor_matrix_3)


    ## TESTING STRASSEN (CORRECTED MATRIX3 c11[3] = 1 instead of 0):
    # factor_matrix_1 = np.array([
    #     [ 1,  0,  1,  0,  1, -1,  0],  # a11
    #     [ 0,  0,  0,  0,  1,  0,  1],  # a12
    #     [ 0,  1,  0,  0,  0,  1,  0],  # a21
    #     [ 1,  1,  0,  1,  0,  0, -1],  # a22
    # ], dtype=float)

    # factor_matrix_2 = np.array([
    #     [ 1,  1,  0, -1,  0,  1,  0],  # b11
    #     [ 0,  0,  1,  0,  0,  1,  0],  # b12
    #     [ 0,  0,  0,  1,  0,  0,  1],  # b21
    #     [ 1,  0, -1,  0,  1,  0,  1],  # b22
    # ], dtype=float)

    # factor_matrix_3 = np.array([
    #     [ 1,  0,  0,  1, -1,  0,  1],  # c11  ### fourth entry is 1 instead of 0. 
    #     [ 0,  0,  1,  0,  1,  0,  0],  # c12 
    #     [ 0,  1,  0,  1,  0,  0,  0],  # c21 
    #     [ 1, -1,  1,  0,  0,  1,  0],  # c22
    # ], dtype=float)

    print(f"\n{'='*60}")
    print(f"  RANK SUMMARY")
    print(f"  Problem : {thisN}x{thisM} × {thisM}x{thisP}")
    print(f"  Rank R  : {thisR}  (= {thisR} scalar multiplications)")
    print(f"  Gene    : {gene_id}")
    print(f"{'='*60}\n")
    
    print(f"Factor shapes: {factor_matrix_1.shape}, {factor_matrix_2.shape}, {factor_matrix_3.shape}")
    print("Factor matrix 1 (U):\n", factor_matrix_1)
    print("Factor matrix 2 (V):\n", factor_matrix_2)
    print("Factor matrix 3 (W):\n", factor_matrix_3)

    U_ref = factor_matrix_1
    V_ref = factor_matrix_2
    W_ref = factor_matrix_3

    # w_c11 = solve_W_row_for_entry(U_ref, V_ref, 0)
    # w_c12 = solve_W_row_for_entry(U_ref, V_ref, 1)
    # w_c21 = solve_W_row_for_entry(U_ref, V_ref, 2)
    # w_c22 = solve_W_row_for_entry(U_ref, V_ref, 3)
    # W_fixed = np.vstack([w_c11, w_c12, w_c21, w_c22])
    # W_fixed_rounded = np.rint(W_fixed).astype(int)
    # print("W from solving linear system:")
    # print(W_fixed_rounded)
    # print("The W i have already:")
    # print(W_ref)
    # print("EQUALITY? ", np.array_equal(W_fixed_rounded, W_ref))
    # # testing a manual W passed in
    # W_ref = W_fixed_rounded



    print(f"\n\n************************************ FULL EVAL PIPELINE START (R={thisR}) ************************************\n")
    print(f"    Rank being evaluated: {thisR}")
    print(f"    This means the algorithm performs exactly {thisR} scalar multiplications to multiply a {thisN}x{thisM} by {thisM}x{thisP} matrix.\n")
    # pipeline_scores = verify_decomposition_pipeline(U_ref, V_ref, W_ref, n=thisN, m=thisM, p=thisP, num_random_tests=10)
    pipeline_scores = verify_decomposition_pipeline(U_ref, V_ref, W_ref, n=thisN, m=thisM, p=thisP, num_random_tests=10)
    print("\n************************************ FULL EVAL PIPELINE END ************************************\n\n")
    
    fitness = pipeline_scores
    fitness_0, fitness_1, fitness_2, fitness_3, fitness_4, fitness_5 = fitness

    # ========================================================================
    # CALCULATE FITNESS
    # ========================================================================
    print("==================================== [ FITNESS ] ====================================")
    print(f"Rank R (scalar multiplications used)  : {thisR}")
    print(f"Fraction of literal matrix entries WRONG (0 is identical match): {fitness_0:.4f}")
    print(f"Magnitude of errors (lower is better): {fitness_1:.4f}")
    print(f"Basis score ratio (higher is better): {fitness_2:.4f}")
    print(f"Random multiplication score (higher is better): {fitness_3:.4f}")
    print(f"Median wall-clock time per multiply (lower is better): {fitness_4} ns")
    print(f"Total additions (lower is better): {fitness_5}")
    #if we want to put rank into the fitness tuple for LLMGE to use in selection, we can do it like this:
    #results_text = f"R={thisR}, {fitness_0:.4f}, {fitness_1:.4f}, {fitness_2:.4f}, {fitness_3:.4f}, {fitness_4:.4f}, {fitness_5:.4f}"
    
    results_text = f" {fitness_0:.4f}, {fitness_1:.4f}, {fitness_2:.4f}, {fitness_3:.4f}, {fitness_4:.4f}, {fitness_5:.4f}"
    print("="*120)

    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    dir_path = os.path.dirname(filename)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(filename, 'w') as file:
        file.write(results_text)

    # Dedicated rank log - prepared for future dynamic rank system
    rank_log = os.path.abspath(f'results/{gene_id}_rank.txt')
    with open(rank_log, 'w') as f:
        f.write(f"gene_id: {gene_id}\n")
        f.write(f"problem: {thisN}x{thisM} × {thisM}x{thisP}\n")
        f.write(f"rank_used: {thisR}\n")
        f.write(f"basis_score: {fitness_2:.4f}\n")
        f.write(f"random_score: {fitness_3:.4f}\n")
        f.write(f"valid: {fitness_2 == 1.0 and fitness_3 == 1.0}\n")
    
    print(f"({results_text})")
    print(f"\nResults written to {filename}")
    print('='*120)
    print('job done')
    print('='*120)