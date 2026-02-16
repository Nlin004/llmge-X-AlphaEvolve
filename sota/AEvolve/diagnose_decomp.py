"""
Direct test of the saved factors from exp2
"""

import numpy as np

print("="*80)
print("DIRECT TEST OF SAVED FACTORS")
print("="*80)

# Load factors
factors_file = "trained/exp4/factors.npz"
print(f"\nLoading: {factors_file}")
factors_data = np.load(factors_file)

U = factors_data['U']
V = factors_data['V']
W = factors_data['W']

print(f"U shape: {U.shape}")
print(f"V shape: {V.shape}")
print(f"W shape: {W.shape}")

# Round to see the pattern
U_rounded = np.round(U)
V_rounded = np.round(V)
W_rounded = np.round(W)

print("\nU (rounded):")
print(U_rounded)
print("\nV (rounded):")
print(V_rounded)
print("\nW (rounded):")
print(W_rounded)

# Test 1: Reconstruct the tensor
print("\n" + "="*80)
print("TEST 1: Tensor Reconstruction")
print("="*80)

reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
reconstructed_rounded = np.rint(reconstructed).astype(np.int32)

print(f"Reconstructed tensor shape: {reconstructed.shape}")
print(f"Non-zero entries: {np.sum(np.abs(reconstructed_rounded) > 0)}")

# Create correct tensor
n = 2
dim = 4
correct_tensor = np.zeros((dim, dim, dim), dtype=np.int32)
for i in range(n):
    for j in range(n):
        for k in range(n):
            a_idx = i * n + j
            b_idx = j * n + k
            c_idx = i * n + k
            correct_tensor[a_idx, b_idx, c_idx] = 1

matches = np.array_equal(reconstructed_rounded, correct_tensor)
print(f"\nTensor matches expected: {matches}")

if not matches:
    diff = reconstructed_rounded - correct_tensor
    num_diff = np.sum(np.abs(diff) > 0)
    print(f"Number of differences: {num_diff}")

# Test 2: Use for matrix multiplication
print("\n" + "="*80)
print("TEST 2: Matrix Multiplication")
print("="*80)

A = np.array([[1.0, 2.0],
              [3.0, 4.0]])

B = np.array([[5.0, 6.0],
              [7.0, 8.0]])

C_expected = A @ B

print("A =")
print(A)
print("\nB =")
print(B)
print("\nExpected C = A @ B =")
print(C_expected)

# Apply decomposition
A_flat = A.flatten()
B_flat = B.flatten()
C_flat = np.zeros(4)

R = U.shape[1]
for r in range(R):
    u_r = U[:, r]
    v_r = V[:, r]
    w_r = W[:, r]
    
    Au = np.dot(u_r, A_flat)
    Bv = np.dot(v_r, B_flat)
    scalar = Au * Bv
    
    C_flat += scalar * w_r

C_result = C_flat.reshape(2, 2)

print("\nComputed C =")
print(C_result)

error = np.max(np.abs(C_result - C_expected))
print(f"\nMax error: {error:.2e}")

if error < 1e-6:
    print("✓ CORRECT!")
else:
    print("✗ WRONG!")
    print("\nDifference:")
    print(C_result - C_expected)

# Test 3: Show what each rank-1 component does
print("\n" + "="*80)
print("TEST 3: Breaking down each rank-1 component")
print("="*80)

print("\nFor A =")
print(A)
print("and B =")
print(B)
print()

A_flat = A.flatten()
B_flat = B.flatten()

for r in range(R):
    u_r = U[:, r]
    v_r = V[:, r]
    w_r = W[:, r]
    
    Au = np.dot(u_r, A_flat)
    Bv = np.dot(v_r, B_flat)
    scalar = Au * Bv
    contribution = scalar * w_r
    
    print(f"Rank-1 component {r}:")
    print(f"  u = {np.round(u_r, 1)}")
    print(f"  v = {np.round(v_r, 1)}")
    print(f"  w = {np.round(w_r, 1)}")
    print(f"  u·A_flat = {Au:.2f}")
    print(f"  v·B_flat = {Bv:.2f}")
    print(f"  scalar = {scalar:.2f}")
    print(f"  contribution to C_flat = {scalar:.2f} * w = {np.round(contribution, 2)}")
    print()

# Test 4: Try a few random matrices
print("="*80)
print("TEST 4: Random matrices")
print("="*80)

num_correct = 0
for test_num in range(10):
    A_test = np.random.randn(2, 2)
    B_test = np.random.randn(2, 2)
    C_expected_test = A_test @ B_test
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    C_flat = np.zeros(4)
    
    for r in range(R):
        C_flat += np.dot(U[:, r], A_flat) * np.dot(V[:, r], B_flat) * W[:, r]
    
    C_result = C_flat.reshape(2, 2)
    error = np.max(np.abs(C_result - C_expected_test))
    
    if error < 1e-6:
        num_correct += 1
    else:
        if test_num == 0:
            print(f"\nTest {test_num} FAILED:")
            print(f"A = \n{A_test}")
            print(f"B = \n{B_test}")
            print(f"Expected C = \n{C_expected_test}")
            print(f"Computed C = \n{C_result}")
            print(f"Error: {error:.2e}")

print(f"\n{num_correct}/10 tests passed")

if num_correct == 0:
    print("\n" + "="*80)
    print("DIAGNOSIS: All tests failed!")
    print("="*80)
    print("\nThe tensor reconstruction matches, but matrix multiplication fails.")
    print("This means the issue is in HOW we're applying the decomposition.")
    print("\nPossible issues:")
    print("1. The einsum interpretation is wrong")
    print("2. The way we flatten/unflatten matrices is inconsistent")
    print("3. The tensor uses a different convention than we expect")