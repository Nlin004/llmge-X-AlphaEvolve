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


def exact_tensor_check(U, V, W, n): # the input for these is done after the projection (either method above, full ints or full+half ints))
    # here we assume U V W factors HAVE been rounded, to make a more "mathematical" tensor that could actually represent product
    dim = n * n

    T_true = np.zeros((dim, dim, dim), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # T_true[i*n+j, j*n+k, k*n+i] = 1
                T_true[i*n+j, j*n+k, i*n+k] = 1
    # RECONSTRUCTION of tensor from U V W.

    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)

    print("\nTarget tensor (T_true):")
    print(T_true)
    print("\nReconstructed tensor (T_hat):")
    print(T_hat)
    # print("Reconstructed but rounded:") # THIS IS NOT CORRECT BECAUSE YOU CANT JUST SNAP THE TENSOR TO CORRECT ONE IF ITS "CLOSE ENOUGH"
    # # constructed_tensor = np.rint(constructed_tensor).astype(np.int32)
    # print(np.rint(T_hat).astype(np.int32))

    # Count incorrect entries
    total_entries = T_true.size
    incorrect = np.sum(~np.isclose(T_hat, T_true, atol=1e-10))
    fraction_incorrect = incorrect / total_entries

    print(f"\nfrac: {incorrect}/{total_entries}")

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
    n = A.shape[0]
    
    # ROW-MAJOR (default) - matches your Strassen factors
    A_flat = A.flatten()  # NO order='F'!
    B_flat = B.flatten()  # NO order='F'!
    C_flat = np.zeros(n * n)
    
    R = U.shape[1]
    
    for r in range(R):
        a_r = np.dot(U[:, r], A_flat)
        b_r = np.dot(V[:, r], B_flat)
        C_flat += (a_r * b_r) * W[:, r]
    
    return C_flat.reshape(n, n)  # NO order='F'!

    # n = A.shape[0]
    # dim = n * n
    # R = U.shape[1]

    # A_flat = A.reshape(-1)
    # B_flat = B.reshape(-1)

    # C_flat = np.zeros(dim, dtype=np.int64)

    # for r in range(R):
    #     a_r = np.dot(U[:, r], A_flat)
    #     b_r = np.dot(V[:, r], B_flat)
    #     C_flat += W[:, r] * (a_r * b_r)

    # print("Done applying decomposition for bilinear:\n")
    # print(C_flat.reshape(n, n, order="F"))
    # return C_flat.reshape(n, n, order="F")
# def check_strassen_basis(U, V, W):
#     n = 2
#     ok = True
#     for i in range(n):
#         for j in range(n):
#             A = np.zeros((n,n), dtype=int)
#             A[i,j] = 1
#             for k in range(n):
#                 for l in range(n):
#                     B = np.zeros((n,n), dtype=int)
#                     B[k,l] = 1

#                     C_expected = A @ B
#                     C_actual = apply_decomposition_bilinear(A,B,U,V,W)

#                     if not np.array_equal(C_actual, C_expected):
#                         print("FAIL at A[{},{}], B[{},{}]".format(i,j,k,l))
#                         print("A:\n", A)
#                         print("B:\n", B)
#                         print("C_expected:\n", C_expected)
#                         print("C_actual:\n", C_actual)
#                         ok = False
#     print("All basis tests pass?" , ok)
#     return ok


# ============ BASIS COMPARISON TESTS =============

def basis_verification_cleaned(U, V, W, n):
    print("**** BASIS TESTS **** ")
    total_tests = n * n * n * n
    counter = 0

    for i in range(n):
        for j in range(n):
            A = np.zeros((n, n), dtype=np.int64)
            A[i, j] = 1

            for k in range(n):
                for l in range(n):
                    B = np.zeros((n, n), dtype=np.int64)
                    B[k, l] = 1

                    C_expected = A @ B
                    C_actual = apply_decomposition_bilinear(A, B, U, V, W).astype(np.int64)

                    print("\nC_expected:\n", C_expected)
                    print("C_actual:\n", C_actual)
                    print("-"*20)

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

def random_matrix_verification(U, V, W, n, num_tests=50, seed=0):
    rng = np.random.default_rng(seed)
    print("**** RANDOM MATRIX TESTS ****")
    counter = 0
    for _ in range(num_tests):
        A = rng.integers(-3, 4, size=(n, n), dtype=np.int64)
        B = rng.integers(-3, 4, size=(n, n), dtype=np.int64)

        C_expected = A @ B
        C_actual = apply_decomposition_bilinear(A, B, U, V, W)

        print("C_expected:\n", C_expected)
        print("C_actual:\n", C_actual)
        print("-"*20)

        if np.array_equal(C_actual, C_expected):
            counter += 1
    print(f"Random matrix verification score: {counter}/{num_tests} tests correct.")
    return counter/num_tests # float


# ============ MAIN VERIFICATION PIPELINE =============

def verify_decomposition_pipeline(U_float, V_float, W_float, n, num_random_tests=50):
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
    ratio_matrix_entries_wrong, magnitude_of_errors = exact_tensor_check(U_half, V_half, W_half, n)
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
    basis_score = basis_verification_cleaned(U_half, V_half, W_half, n)
    # basis_score = basis_verification_vs_ref(U_half, V_half, W_half, U_ref, V_ref, W_ref, n) # using T_ref instead of canonical A@B as the basis for correctness, since the model is really trying to match T_ref not necessarily the canonical tensor.


    # Step 3: random tests
    random_score = random_matrix_verification(U_half, V_half, W_half, n, num_random_tests)
    # random_score = random_matrix_verification_vs_ref(U_half, V_half, W_half, U_ref, V_ref, W_ref, n, num_random_tests) # using T_ref instead of canonical A@B as the basis for correctness, since the model is really trying to match T_ref not necessarily the canonical tensor.

    # Done: return all metrics gathered from the suites of tests as a 4-tuple for LLMGE 
    fitness = (ratio_matrix_entries_wrong, magnitude_of_errors, basis_score, random_score)
    return fitness



def compute_reference_tensor(U_ref, V_ref, W_ref):
    """
    Given reference factors (U_ref, V_ref, W_ref) of shape (n^2, R),
    compute and return the reference tensor T_ref with shape (n^2, n^2, n^2).
    """
    T_ref = np.einsum('ir,jr,kr->ijk', U_ref, V_ref, W_ref)
    return T_ref













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









# BELOW 2 FUNCTIONS ARE FOR CHECKING TENSOR REPRESENTATION CORRECTNESS.

# # ===================== Helper to generate the matmul tensor =====================

# def generate_matmul_tensor(n):
#     dim = n * n
#     T = np.zeros((dim, dim, dim), dtype=np.int32)

#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 T[i*n + j, j*n + k, k*n + i] = 1

#     return T

# # ===================== VERIFICATION VIA *TENSOR DECOMP* CORRECTNESS: =====================

# def fundamental_correctness_check(decomposition, n, m, p_dim):
#     """
#     EXACT AlphaEvolve tensor verification.
#     """
#     U, V, W = decomposition

#     matmul_tensor = np.zeros((n*m, m*p_dim, p_dim*n), dtype=np.int32)
#     for i in range(n):
#         for j in range(m):
#             for k in range(p_dim):
#                 matmul_tensor[i*m + j, j*p_dim + k, k*n + i] = 1
#     # matmul_tensor = generate_matmul_tensor(n)

#     constructed_tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
#     # constructed_tensor = np.rint(constructed_tensor).astype(np.int32)
#     constructed_tensor_complete = np.rint(constructed_tensor).astype(np.int32)
#     # DO NOT ROUND!!!!
#     # err = np.max(np.abs(constructed_tensor - matmul_tensor))
#     # if err > 1e-8:
#     #     return False

#     print("TRUTH:\n")
#     print(matmul_tensor)
#     print("Not Rounded:\n")
#     print(constructed_tensor)
#     print("Rounded:\n")
#     print(constructed_tensor_complete)

#     return np.array_equal(constructed_tensor_complete, matmul_tensor)





# def apply_decomposition(A, B, U, V, W):
#     """
#     Apply bilinear algorithm defined by (U, V, W).
#     """
#     n = A.shape[0]
#     A_flat = A.reshape(-1)
#     B_flat = B.reshape(-1)
#     C_flat = np.zeros(n * n)

#     R = U.shape[1]

#     for r in range(R):
#         a_r = np.dot(U[:, r], A_flat)
#         b_r = np.dot(V[:, r], B_flat)
#         C_flat += (a_r * b_r) * W[:, r]

#     return C_flat.reshape(n, n, order="F") # ensure column-major order!!!!!

    
# def basis_verification(U, V, W, n, atol=1e-5):
#     """
#     Verify decomposition on all n^2 by n^2 basis matrix pairs.
#     Prints detailed comparison for each test.
#     """

#     # Round to half-integers (AlphaEvolve approach)
#     # U = np.round(U * 2) / 2
#     # V = np.round(V * 2) / 2
#     # W = np.round(W * 2) / 2



#     total_tests = n * n * n * n
#     passed_tests = 0
#     failed_tests = []
    
#     print("\n" + "="*100)
#     print(f"BASIS VERIFICATION: Testing all {total_tests} basis matrix pairs for {n}×{n} matrices")
#     print("="*100)
    
#     test_num = 0
    
#     for i in range(n):
#         for j in range(n):
#             # Create basis matrix A with single 1 at position (i,j)
#             A = np.zeros((n, n))
#             A[i, j] = 1.0
            
#             for k in range(n):
#                 for l in range(n):
#                     test_num += 1
                    
#                     # Create basis matrix B with single 1 at position (k,l)
#                     B = np.zeros((n, n))
#                     B[k, l] = 1.0
                    
#                     # Expected result
#                     C_expected = A @ B
                    
#                     # Actual result from decomposition
#                     # C_actual = apply_decomposition(A, B, U, V, W)  # Use ROUNDED factors?
#                     C_actual = apply_decomposition_to_multiply(A, B, U, V, W)  # Use original factors without rounding for actual multiplication result
                    
                    
#                     # Calculate error
#                     error_matrix = C_actual - C_expected
#                     max_error = np.max(np.abs(error_matrix))
                    
#                     # Check if passes
#                     passes = np.allclose(C_actual, C_expected, atol=atol)
                    
#                     if passes:
#                         passed_tests += 1
#                     else:
#                         failed_tests.append((test_num, i, j, k, l, max_error))
                    
#                     # Print detailed comparison
#                     status = "PASS" if passes else "FAIL"
#                     # print(f"\nTest {test_num}/{total_tests}: A[{i},{j}]=1 × B[{k},{l}]=1  →  {status}")
#                     # print(f"{'─'*100}")
                    
#                     # Side-by-side comparison
#                     # print(f"Expected C = A @ B:          Actual C (from decomposition):")
#                     for row_idx in range(n):
#                         expected_row = "  ".join([f"{C_expected[row_idx, col_idx]:7.4f}" for col_idx in range(n)])
#                         actual_row = "  ".join([f"{C_actual[row_idx, col_idx]:7.4f}" for col_idx in range(n)])
#                         # print(f"  [{expected_row}]    [{actual_row}]")
                    
#                     # Error matrix
#                     # print(f"\nError = Actual - Expected:")
#                     for row_idx in range(n):
#                         error_row = "  ".join([f"{error_matrix[row_idx, col_idx]:+7.4f}" for col_idx in range(n)])
#                         # print(f"  [{error_row}]")
                    
#                     # print(f"Max error: {max_error:.6e}  (tolerance: {atol:.6e})")
                    
#                     # if not passes:
#                     #     print(f"FAILED: Error {max_error:.6e} exceeds tolerance {atol:.6e}")
    
    # Summary
    # print("\n" + "="*100)
    # print("VERIFICATION SUMMARY")
    # print("="*100)
    # print(f"Total tests: {total_tests}")
    # print(f"Passed: {passed_tests}/{total_tests} ({100*passed_tests/total_tests:.1f}%)")
    # print(f"Failed: {len(failed_tests)}/{total_tests} ({100*len(failed_tests)/total_tests:.1f}%)")
    # ratio = passed_tests / total_tests

    # if failed_tests:
    #     print(f"\nVERIFICATION FAILED")
    #     print(f"\nFailed tests (first 10):")
    #     for test_num, i, j, k, l, max_error in failed_tests[:10]:
    #         print(f"  Test {test_num}: A[{i},{j}]=1 × B[{k},{l}]=1  →  Max error: {max_error:.6e}")
        
    #     if len(failed_tests) > 10:
    #         print(f"  ... and {len(failed_tests) - 10} more failures")
    
    # else:
    #     print(f"VERIFICATION PASSED - All {total_tests} basis tests correct!")

    # # for LLMGE: return the ratio of passed tests as a fitness metric (higher is better), so like 14/16 means only a couple failed.
    # return ratio



# ===================== HELPER FUNCTION TO CONVERT TENSOR FACTORS TO MULT: =====================

# def apply_decomposition_to_multiply(A, B, factor_matrix_1, factor_matrix_2, factor_matrix_3):
#     """
#     Apply tensor decomposition to multiply matrices A and B.
    
#     The decomposition represents the matrix multiplication tensor.
#     For <n,n,n> multiplication, the factors are (n²×R) matrices.
    
#     Key insight: Each column r in the factors represents one "multiplication":
#     - factor_matrix_1[:, r] selects elements from A (flattened)
#     - factor_matrix_2[:, r] selects elements from B (flattened)
#     - factor_matrix_3[:, r] determines where the product goes in C (flattened)
    
#     Args:
#         A: Matrix of shape (n, n)
#         B: Matrix of shape (n, n)
#         factor_matrix_1: Shape (n^2, R)
#         factor_matrix_2: Shape (n^2, R)
#         factor_matrix_3: Shape (n^2, R)
    
#     Returns:
#         C: Result matrix of shape (n, n)
#     """
#     n = A.shape[0]
#     R = factor_matrix_1.shape[1]  # Number of rank-1 components (multiplications)
    
#     # Flatten input matrices
#     A_flat = A.flatten()  # Shape: (n2,)
#     B_flat = B.flatten()  # Shape: (n2,)
#     C_flat = np.zeros(n * n)  # Shape: (n2,)
    
#     # Apply each rank-1 component
#     for r in range(R):
#         u = factor_matrix_1[:, r]  # Select from A
#         v = factor_matrix_2[:, r]  # Select from B
#         w = factor_matrix_3[:, r]  # Contribute to C
        
#         # One scalar multiplication
#         Au = np.dot(u, A_flat)
#         Bv = np.dot(v, B_flat)
#         scalar = Au * Bv
        
#         # Add contribution to result
#         C_flat += scalar * w

#     C = C_flat.reshape(n, n, order="F")  #ROW MAJOR ORDER!!!!!
#     return C

# # ===================== VERIFICATION VIA RANDOM MULTIPLICATION TESTS: =====================
# def verify_decomposition(factor_matrix_1, factor_matrix_2, factor_matrix_3, 
#                          n=2, num_tests=50, tol=2e-1):
#     """
#     Verify that the decomposition correctly multiplies matrices.
    
#     Tests on random matrices and returns detailed statistics.
    
#     Args:
#         factor_matrix_1, factor_matrix_2, factor_matrix_3: Decomposition factors
#         n: Matrix dimension
#         num_tests: Number of random tests
#         tol: Error tolerance
    
#     Returns:
#         correct_ratio: Fraction of tests passed (0.0 to 1.0)
#         num_correct: Number of tests passed
#         num_total: Total tests
#         max_error: Maximum error observed
#         avg_error: Average error
#         num_multiplications: Rank of decomposition
#     """
#     num_correct = 0
#     max_error = 0.0
#     total_error = 0.0
    
#     # Count multiplications (rank of decomposition)
#     num_multiplications = factor_matrix_1.shape[1]
    
#     for _ in range(num_tests):
#         # Generate random test matrices
#         A = np.random.randn(n, n)
#         B = np.random.randn(n, n)
        
#         # Ground truth
#         C_expected = A @ B

#         # print("\nGROUND TRUTH MULT")
#         # print(C_expected)
        
#         # print("\nACTUAL PRODUCT:")
#         # Use decomposition
#         C_result = apply_decomposition_to_multiply(
#             A, B, factor_matrix_1, factor_matrix_2, factor_matrix_3
#         )
        
#         # print(C_result)
#         # Compute error
#         error = np.max(np.abs(C_result - C_expected))
#         max_error = max(max_error, error)
#         total_error += error
        
#         if error <= tol:
#             num_correct += 1
    
#     correct_ratio = num_correct / num_tests
#     avg_error = total_error / num_tests
#     print(f"Avg error over {num_tests} tests: {avg_error:.2e}, Max error: {max_error:.2e}")
#     return correct_ratio, num_correct, num_tests, max_error, avg_error, num_multiplications


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
        # model_module.main()
        old_argv = sys.argv
        # sys.argv = [
        #     old_argv[0],
        #     "--save_dir", str(run_dir),
        #     "--N", str(args.N),
        #     "--R", str(args.R),
        # ]
        sys.argv = [
            old_argv[0],
            "--save_dir", str(run_dir),
        ]


        # Make sure we're in the right directory
        os.chdir(script_directory)

        
        model_module.main()

        
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
        print(f"\nLoaded factors from {factors_file}")

    except Exception as e:
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
    thisN = int(factors_data['N'])
    thisTargetRank = int(factors_data["R"])

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
    
    print(f"Factor shapes: {factor_matrix_1.shape}, {factor_matrix_2.shape}, {factor_matrix_3.shape}")
    print("Factor matrix 1 (U):\n", factor_matrix_1)
    print("Factor matrix 2 (V):\n", factor_matrix_2)
    print("Factor matrix 3 (W):\n", factor_matrix_3)

    # U_ref = np.rint(factor_matrix_1).astype(int)  # known good U from strassen
    # V_ref = np.rint(factor_matrix_2).astype(int)  # known good V from strassen
    # W_ref = np.rint(factor_matrix_3).astype(int)  # known good W from strassen
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



    print("\n\n************************************ FULL EVAL PIPELINE START ************************************\n")
    pipeline_scores = verify_decomposition_pipeline(U_ref, V_ref, W_ref, n=thisN, num_random_tests=10)
    print("\n************************************ FULL EVAL PIPELINE END ************************************\n\n")
    
    fitness = pipeline_scores
    fitness_0, fitness_1, fitness_2, fitness_3 = fitness
    # ============= USE THIS FOR VERIFICATION VIA RANDOM MATRIX MULT CHECKS! ============
    # correct_ratio, num_correct, num_total, max_error, avg_error, num_multiplications = verify_decomposition(
    #     factor_matrix_1, factor_matrix_2, factor_matrix_3,
    #     n=thisN,
    #     num_tests=args.num_tests
    # )    
    # # print("PERCENTAGE CORRECT!!!!\n")
    # # print(correct_ratio)
    # standard_mults = thisN ** 3  # Standard algorithm for n×n matrices
    
    # ========================================================================
    # CALCULATE FITNESS
    # ========================================================================
    ONE_FITNESS = False
    # if ONE_FITNESS:

    #     # use is_correct from tensor verification for a simple fitness function
        
    #     if correct_ratio == 1.0:
    #         # Perfect: optimize for performance
    #         # Scale num_multiplications to be comparable to small differences in ratio
    #         fitness = num_multiplications * 1000
    #         status = "FULLY CORRECT"
            
    #     elif correct_ratio >= 0.8:
    #         # Mostly correct: medium penalty
    #         penalty = (1.0 - correct_ratio) * 100000
    #         fitness = penalty + num_multiplications * 1000 + avg_error * 10000
    #         status = f"MOSTLY CORRECT ({correct_ratio:.0%})"
            
    #     else:
    #         # Mostly wrong: large penalty
    #         penalty = (1.0 - correct_ratio) * 1000000
    #         fitness = penalty + avg_error * 10000 + num_multiplications
    #         status = f"INCORRECT ({correct_ratio:.0%})"
    # else:
    #     # Multi-objective: return tuple (lower is better for both)
    #     fitness_0 = basis_ratio
    #     fitness_1 = correct_ratio  # Maximize correct problems solved (1.0 = perfect, 0.0 = all wrong)
    #     fitness_2 = num_multiplications  # Minimize rank (7 = target)
    #     fitness = (fitness_0, fitness_1, fitness_2)
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    
    
    # ===========================
    # print(f"\nCorrectness (using seed model's verification):")
    # if is_correct:
    #    fitness = num_multiplications * 1000  # e.g., 7000 for rank-7
    # else:
    #    fitness = 1000000 + num_multiplications  # e.g., 1000007 if wrong
    # ===========================

    # ============================
    
    print(f"\nFitness for LLMGE:")
    
    if ONE_FITNESS:
        if correct_ratio == 1.0:
            print(f"  Breakdown: {num_multiplications} mults × 1000")
        elif correct_ratio >= 0.8:
            penalty = (1.0 - correct_ratio) * 100000
            print(f"  Breakdown: {penalty:.2f} (penalty) + {num_multiplications * 1000} (mults) + {avg_error * 10000:.2f} (error)")
        else:
            penalty = (1.0 - correct_ratio) * 1000000
            print(f"  Breakdown: {penalty:.2f} (penalty) + {avg_error * 10000:.2f} (error) + {num_multiplications} (mults)")

        print(f"\n{fitness:.4f}")
        # print(f"# Breakdown: ratio={correct_ratio:.4f}, rank={num_multiplications}, error={avg_error:.2e}")
        
        # Save results
        results_text = f"{fitness:.4f}"
    else: 
        print("=====================================================================")
        print(f"Fraction of literal matrix entries WRONG (0 is identical match): {fitness_0:.4f}")
        print(f"Magnitude of errors (lower is better): {fitness_1:.4f}")
        print(f"Basis score ratio (higher is better): {fitness_2:.4f}")
        print(f"Random multiplication score (higher is better): {fitness_3:.4f}")
        results_text = f"{fitness_0:.4f}, {fitness_1:.4f}, {fitness_2:.4f}, {fitness_3:.4f}"

        
    print("="*120)

    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    dir_path = os.path.dirname(filename)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(filename, 'w') as file:
        file.write(results_text)
    
    print(results_text)
    print(f"\nResults written to {filename}")
    print('='*120)
    print('job done')
    print('='*120)