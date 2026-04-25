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


# =============================================================================
# DYNAMIC RANK SEARCH CONFIGURATION
# Edit this list to control which matrix sizes and rank ranges are searched.
# For each size, the evaluator descends from starting_R until no valid solution
# is found. trivial_rank = N*M*P is computed inline (not stored here).
# =============================================================================
MATRIX_SEARCH_CONFIGS = [
    {'N': 3, 'M': 3, 'P': 3, 'starting_R': 24, 'min_R': 22},
    # Add more sizes here, e.g.:
    # {'N': 2, 'M': 3, 'P': 2, 'starting_R': 11, 'min_R': 8},
    # {'N': 3, 'M': 2, 'P': 3, 'starting_R': 15, 'min_R': 11},
]
# Iterations and early-stop threshold passed to each model run during rank search.
RANK_SEARCH_ITERATIONS = 10000
RANK_SEARCH_EARLY_STOP = 0
# =============================================================================


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


# # Is the tensor from (U,V,W) exactly equal to the tensor from (U_ref,V_ref,W_ref) in this basis?
# def exact_tensor_check_against_ref(U, V, W, T_ref):
#     """
#     Check if candidate (U,V,W) reproduces the same tensor as T_ref.
#     """
#     print("Reference tensor (T_ref):")
#     print(T_ref)
#     T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
#     # If everything is integral/half-integral, this can be exact equality.
#     print("Reconstructed tensor (T_hat):")
#     print(T_hat)
#     return np.array_equal(T_hat, T_ref)
#     # define “correctness” as “equivalent to the reference algorithm,” not necessarily the canonical A @ B with the chosen flattening.

#     # doing the T_ref to check -> does the candidate reconstruct the same tensor as this possibly wrong Strassen variant

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


# def apply_reference(A, B, U_ref, V_ref, W_ref):
#     return apply_decomposition_bilinear(A, B, U_ref, V_ref, W_ref)
# def basis_verification_vs_ref(U, V, W, U_ref, V_ref, W_ref, n):
#     total = n*n*n*n
#     counter = 0
#     for i in range(n):
#         for j in range(n):
#             A = np.zeros((n,n), dtype=int)
#             A[i,j] = 1
#             for k in range(n):
#                 for l in range(n):
#                     B = np.zeros((n,n), dtype=int)
#                     B[k,l] = 1
#                     C_expected = apply_reference(A,B,U_ref,V_ref,W_ref)
#                     C_actual   = apply_decomposition_bilinear(A,B,U,V,W)
#                     if np.array_equal(C_actual, C_expected):
#                         counter += 1
#     return counter / total



# ============ RANDOM MATRIX TESTS =============

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
    
    # run_dir is now created per (size, rank) attempt inside the search loop below.





    print("="*120)
    print(f"Gene ID: {gene_id}")
    print(f"Matrix sizes to search: {[(c['N'], c['M'], c['P']) for c in MATRIX_SEARCH_CONFIGS]}")
    print("="*120)

    # ========================================================================
    # DYNAMIC RANK SEARCH — iterate over sizes, descend rank until failure
    # ========================================================================
    all_size_results = []

    for size_cfg in MATRIX_SEARCH_CONFIGS:
        N, M, P = size_cfg['N'], size_cfg['M'], size_cfg['P']
        trivial_rank = N * M * P
        best_valid_R = None
        best_valid_fitness_tuple = None

        print(f"\n{'='*80}")
        print(f"  Searching {N}x{M} × {M}x{P}  |  trivial_rank={trivial_rank}  |  "
              f"R range: {size_cfg['starting_R']} -> {size_cfg['min_R']}")
        print(f"{'='*80}")

        # best_attempt tracks the most recent real metrics (even if invalid),
        # so LLMGE gets gradient signal rather than pure sentinels when no
        # valid solution is found.
        best_attempt_fitness_tuple = None

        for R in range(size_cfg['starting_R'], size_cfg['min_R'] - 1, -1):
            run_dir = (p(args.save_dir).resolve() /
                       f"{gene_id}_N{N}M{M}P{P}_R{R}_pid{os.getpid()}")
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [R={R}] Running model for {N}x{M}x{P}...")
            old_argv = sys.argv
            try:
                sys.argv = [
                    old_argv[0],
                    "--save_dir", str(run_dir),
                    "--N", str(N),
                    "--M", str(M),
                    "--P", str(P),
                    "--R", str(R),
                    "--iterations", str(RANK_SEARCH_ITERATIONS),
                    "--early_stop_threshold", str(RANK_SEARCH_EARLY_STOP),
                ]
                os.chdir(script_directory)
                model_module.main()
                sys.argv = old_argv

                factors_file = run_dir / "factors.npz"
                if not factors_file.exists():
                    print(f"  [R={R}] No factors.npz found — stopping rank descent.")
                    break

                factors_data = np.load(factors_file)

                # Guard: model must respect the CLI dimensions
                fd_N = int(factors_data['N'])
                fd_M = int(factors_data['M'])
                fd_P = int(factors_data['P'])
                if fd_N != N or fd_M != M or fd_P != P:
                    print(f"  [R={R}] Dimension mismatch: expected {N}x{M}x{P}, "
                          f"got {fd_N}x{fd_M}x{fd_P}. Model ignores CLI dims — stopping.")
                    break

                U, V, W = factors_data['U'], factors_data['V'], factors_data['W']

                print(f"\n  [R={R}] **** EVAL PIPELINE START ****")
                print(f"  Rank being evaluated: {R}  "
                      f"({R} scalar multiplications for {N}x{M} x {M}x{P})")
                fitness_tuple = verify_decomposition_pipeline(
                    U, V, W, n=N, m=M, p=P, num_random_tests=10)
                print(f"  [R={R}] **** EVAL PIPELINE END ****\n")

                # Always keep the most recent real metrics as fallback
                best_attempt_fitness_tuple = fitness_tuple

                fundamental_check, _, basis_score, random_score, _, _ = fitness_tuple
                is_valid = (fundamental_check == 0.0 and basis_score == 1.0 and random_score == 1.0)

                if is_valid:
                    print(f"  [R={R}] VALID (basis={basis_score:.4f}, "
                          f"random={random_score:.4f}) — trying R={R-1}")
                    best_valid_R = R
                    best_valid_fitness_tuple = fitness_tuple
                else:
                    print(f"  [R={R}] INVALID (basis={basis_score:.4f}, "
                          f"random={random_score:.4f}) — stopping rank descent.")
                    break

            except Exception as e:
                sys.argv = old_argv
                print(f"  [R={R}] Exception: {e} — stopping rank descent.")
                break

        # Aggregate per-size result.
        # If a valid solution was found, use its rank and metrics.
        # If not, fall back to actual metrics from starting_R (so LLMGE sees
        # real gradient signal, e.g. basis=0.75 not 0.0), but rank_ratio=1.0
        # to penalise the lack of a verified decomposition.
        if best_valid_R is not None:
            ratio_wrong, magnitude_errors, basis_score, random_score, median_ns, additions = best_valid_fitness_tuple
            rank_ratio = best_valid_R / trivial_rank
            magnitude_errors = magnitude_errors if np.isfinite(magnitude_errors) else 1000.0
            median_ns = min(float(median_ns), 1e9)
            additions = min(float(additions), 1e6)
            print(f"\n  Best valid rank for {N}x{M}x{P}: R={best_valid_R} "
                  f"(rank_ratio={rank_ratio:.4f})")
        elif best_attempt_fitness_tuple is not None:
            # Real metrics from the (invalid) starting_R run — preserve gradient signal
            ratio_wrong, magnitude_errors, basis_score, random_score, median_ns, additions = best_attempt_fitness_tuple
            rank_ratio = 1.0  # penalise: no valid solution
            magnitude_errors = magnitude_errors if np.isfinite(magnitude_errors) else 1000.0
            median_ns = min(float(median_ns), 1e9)
            additions = min(float(additions), 1e6)
            print(f"\n  No valid solution for {N}x{M}x{P} — using real metrics from starting_R run.")
        else:
            # Model crashed before producing any metrics
            rank_ratio = 1.0
            ratio_wrong = 1.0
            magnitude_errors = 1000.0
            basis_score = 0.0
            random_score = 0.0
            median_ns = 1e9
            additions = 1e6
            print(f"\n  No valid solution found for {N}x{M}x{P} (sentinel values used).")

        all_size_results.append({
            'N': N, 'M': M, 'P': P,
            'best_valid_R': best_valid_R,
            'rank_ratio': rank_ratio,
            'ratio_wrong': ratio_wrong,
            'magnitude_errors': magnitude_errors,
            'basis_score': basis_score,
            'random_score': random_score,
            'median_ns': median_ns,
            'additions': additions,
        })

    # ========================================================================
    # AGGREGATE FITNESS ACROSS ALL SIZES
    # Fitness tuple (6 values, matching FITNESS_WEIGHTS = (-1,-1,+1,+1,-1,-1)):
    #   [0] avg_rank_ratio  — mean(best_valid_R / N*M*P) across sizes (MINIMIZE)
    #   [1] avg_ratio_wrong — mean fraction of tensor entries wrong (MINIMIZE)
    #   [2] avg_basis_score — mean basis test pass rate (MAXIMIZE)
    #   [3] avg_random_score— mean random test pass rate (MAXIMIZE)
    #   [4] avg_median_ns   — mean wall-clock time per multiply in ns (MINIMIZE)
    #   [5] avg_additions   — mean total addition operations (MINIMIZE)
    # ========================================================================
    avg_rank_ratio  = float(np.mean([r['rank_ratio']   for r in all_size_results]))
    avg_ratio_wrong = float(np.mean([r['ratio_wrong']  for r in all_size_results]))
    avg_basis       = float(np.mean([r['basis_score']  for r in all_size_results]))
    avg_random      = float(np.mean([r['random_score'] for r in all_size_results]))
    avg_ns          = float(np.mean([r['median_ns']    for r in all_size_results]))
    avg_additions   = float(np.mean([r['additions']    for r in all_size_results]))

    print("\n==================================== [ FITNESS SUMMARY ] ====================================")
    for r in all_size_results:
        tag = f"N={r['N']} M={r['M']} P={r['P']}"
        valid_str = f"R={r['best_valid_R']}" if r['best_valid_R'] is not None else "NONE"
        print(f"  {tag}  best_R={valid_str}  rank_ratio={r['rank_ratio']:.4f}  "
              f"basis={r['basis_score']:.4f}  random={r['random_score']:.4f}")
    print(f"\n  avg_rank_ratio  (MINIMIZE): {avg_rank_ratio:.6f}")
    print(f"  avg_ratio_wrong (MINIMIZE): {avg_ratio_wrong:.4f}")
    print(f"  avg_basis_score (MAXIMIZE): {avg_basis:.4f}")
    print(f"  avg_random_score(MAXIMIZE): {avg_random:.4f}")
    print(f"  avg_median_ns   (MINIMIZE): {avg_ns:.4f} ns")
    print(f"  avg_additions   (MINIMIZE): {avg_additions:.4f}")
    print("="*120)

    results_text = (f" {avg_rank_ratio:.6f}, {avg_ratio_wrong:.4f}, "
                    f"{avg_basis:.4f}, {avg_random:.4f}, "
                    f"{avg_ns:.4f}, {avg_additions:.4f}")

    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    dir_path = os.path.dirname(filename)
    os.makedirs(dir_path, exist_ok=True)

    with open(filename, 'w') as file:
        file.write(results_text)

    # Rank log: one entry per searched size
    rank_log = os.path.abspath(f'results/{gene_id}_rank.txt')
    with open(rank_log, 'w') as f:
        f.write(f"gene_id: {gene_id}\n")
        for r in all_size_results:
            f.write(f"size: {r['N']}x{r['M']}x{r['P']}  "
                    f"best_valid_R: {r['best_valid_R']}  "
                    f"rank_ratio: {r['rank_ratio']:.4f}  "
                    f"valid: {r['best_valid_R'] is not None}\n")

    print(f"({results_text})")
    print(f"\nResults written to {filename}")
    print('='*120)
    print('job done')
    print('='*120)