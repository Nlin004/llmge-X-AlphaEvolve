"""
AlphaEvolve Matrix Multiplication Evaluator

reference based on the AlphaEvolve paper: https://arxiv.org/pdf/2506.13131
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MultiplicationResult:
    """Results from a matrix multiplication evaluation"""
    result: np.ndarray
    num_multiplications: int
    execution_time: float
    is_correct: bool
    max_error: float
    method: str


class TensorDecompositionMultiplier:
    """
    Implements matrix multiplication via tensor decomposition.
    
    A matrix multiplication C = A @ B where A is m×n and B is n×p
    can be represented as a rank-R decomposition of a 3D tensor.
    
    The decomposition consists of R triplets (u_i, v_i, w_i) where:
    - u_i has shape (m,)
    - v_i has shape (n,)  
    - w_i has shape (p,)
    
    The multiplication is computed as:
    C[i,j] = sum_r (u_r[i] * v_r @ B @ w_r[j])
    
    More specifically, for each rank-1 component:
    - Compute the scalar: s_r = (u_r^T @ A) @ (v_r) @ (B @ w_r)
    - This requires exactly 1 scalar multiplication
    - Accumulate: C += s_r * (outer product structure)
    """
    
    def __init__(self, m: int, n: int, p: int, decomposition: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None):
        """
        Initialize the multiplier.
        
        Args:
            m: Number of rows in first matrix
            n: Number of columns in first matrix / rows in second matrix
            p: Number of columns in second matrix
            decomposition: List of (u, v, w) triplets forming the tensor decomposition.
                          If None, uses standard algorithm.
        """
        self.m = m
        self.n = n
        self.p = p
        self.decomposition = decomposition
        self.multiplication_count = 0
        
    def multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Multiply matrices A and B using the tensor decomposition.
        
        Args:
            A: Matrix of shape (m, n)
            B: Matrix of shape (n, p)
            
        Returns:
            Tuple of (result matrix, number of scalar multiplications)
        """
        assert A.shape == (self.m, self.n), f"A must be {self.m}×{self.n}, got {A.shape}"
        assert B.shape == (self.n, self.p), f"B must be {self.n}×{self.p}, got {B.shape}"
        
        self.multiplication_count = 0
        
        if self.decomposition is None:
            # Fall back to standard algorithm
            return self._standard_multiply(A, B)
        
        # Flatten input matrices
        A_flat = A.flatten()  # Shape: (m*n,)
        B_flat = B.flatten()  # Shape: (n*p,)
        
        # Initialize result with appropriate dtype
        # Use the dtype that results from multiplying A and B elements
        result_dtype = (A.flat[0] * B.flat[0]).dtype
        C_flat = np.zeros(self.m * self.p, dtype=result_dtype)
        
        # Apply each rank-1 component of the decomposition
        for u, v, w in self.decomposition:
            # Each triplet represents one scalar multiplication
            # u selects from A (length m*n)
            # v selects from B (length n*p)
            # w determines output contribution (length m*p)
            
            # Step 1: Linear combination of A's elements
            Au = np.dot(u, A_flat)  # No scalar multiplications, just additions
            
            # Step 2: Linear combination of B's elements
            Bv = np.dot(v, B_flat)  # No scalar multiplications, just additions
            
            # Step 3: The actual scalar multiplication
            scalar = Au * Bv  # This is the ONE multiplication for this component
            self.multiplication_count += 1
            
            # Step 4: Add contribution to result
            C_flat += scalar * w
        
        # Reshape result back to matrix
        C = C_flat.reshape(self.m, self.p)
        
        return C, self.multiplication_count
    
    def _standard_multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, int]:
        """Standard O(mnp) matrix multiplication for comparison."""
        C = np.zeros((self.m, self.p), dtype=A.dtype)
        mult_count = 0
        
        for i in range(self.m):
            for j in range(self.p):
                for k in range(self.n):
                    C[i, j] += A[i, k] * B[k, j]
                    mult_count += 1
        
        return C, mult_count


def strassen_2x2_decomposition() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Strassen's algorithm for 2×2 matrix multiplication.
    Uses 7 multiplications instead of 8.
    
    Returns rank-7 decomposition for ⟨2,2,2⟩ tensor.
    """
    decomposition = [
        # M1 = (a11 + a22)(b11 + b22)
        (np.array([1., 0., 0., 1.]), 
         np.array([1., 0., 0., 1.]),
         np.array([1., 0., 0., 1.])),
        
        # M2 = (a21 + a22)b11
        (np.array([0., 0., 1., 1.]),
         np.array([1., 0., 0., 0.]),
         np.array([1., 0., 0., 0.])),
        
        # M3 = a11(b12 - b22)
        (np.array([1., 0., 0., 0.]),
         np.array([0., 1., 0., -1.]),
         np.array([0., 1., 0., 0.])),
        
        # M4 = a22(b21 - b11)
        (np.array([0., 0., 0., 1.]),
         np.array([-1., 0., 1., 0.]),
         np.array([0., 0., 0., 1.])),
        
        # M5 = (a11 + a12)b22
        (np.array([1., 1., 0., 0.]),
         np.array([0., 0., 0., 1.]),
         np.array([0., 1., 0., 0.])),
        
        # M6 = (a21 - a11)(b11 + b12)
        (np.array([-1., 0., 1., 0.]),
         np.array([1., 1., 0., 0.]),
         np.array([0., 0., 1., 0.])),
        
        # M7 = (a12 - a22)(b21 + b22)
        (np.array([0., 1., 0., -1.]),
         np.array([0., 0., 1., 1.]),
         np.array([0., 0., 0., 1.])),
    ]
    
    # Note: The above is a simplified representation. The actual Strassen algorithm
    # requires additional linear combinations to reconstruct the result matrix.
    # For a complete implementation, we'd need to specify how to combine the 7 products.
    
    return decomposition


def alphaevolve_4x4_complex_decomposition() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    AlphaEvolve's discovered algorithm for 4×4 complex matrix multiplication.
    Uses 48 multiplications - first improvement over Strassen's recursive application (49) in 56 years!
    
    Note: This is a placeholder. The actual decomposition from the paper would need to be
    loaded from their published results.
    
    Returns rank-48 decomposition for ⟨4,4,4⟩ tensor (complex-valued).
    """
    # This would contain the actual 48 triplets discovered by AlphaEvolve
    # Each triplet (u, v, w) where u, v, w are complex vectors
    # For demonstration, we return None to indicate it's not yet loaded
    return None


def evaluate_multiplication(A: np.ndarray, B: np.ndarray, 
                           decomposition: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
                           method_name: str = "Custom") -> MultiplicationResult:
    """
    Evaluate a matrix multiplication method.
    
    Args:
        A: First matrix
        B: Second matrix
        decomposition: Tensor decomposition to use (None for standard algorithm)
        method_name: Name of the method for reporting
        
    Returns:
        MultiplicationResult with performance metrics
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"Matrix dimensions incompatible: {A.shape} × {B.shape}"
    
    # Ground truth using NumPy
    expected = A @ B
    
    # Create multiplier and run
    multiplier = TensorDecompositionMultiplier(m, n, p, decomposition)
    
    start_time = time.perf_counter()
    result, num_mults = multiplier.multiply(A, B)
    end_time = time.perf_counter()
    
    # Check correctness
    error = np.max(np.abs(result - expected))
    is_correct = error < 1e-10  # Tolerance for numerical errors
    
    return MultiplicationResult(
        result=result,
        num_multiplications=num_mults,
        execution_time=end_time - start_time,
        is_correct=is_correct,
        max_error=error,
        method=method_name
    )


def compare_methods(A: np.ndarray, B: np.ndarray, 
                   decompositions: dict = None) -> None:
    """
    Compare different multiplication methods on the same input.
    
    Args:
        A: First matrix
        B: Second matrix  
        decompositions: Dictionary mapping method names to decompositions
    """
    print(f"\n{'='*80}")
    print(f"Matrix Multiplication Comparison")
    print(f"{'='*80}")
    print(f"A shape: {A.shape}, dtype: {A.dtype}")
    print(f"B shape: {B.shape}, dtype: {B.dtype}")
    print(f"Result shape: {A.shape[0]}×{B.shape[1]}")
    
    # Standard multiplication baseline
    print(f"\n{'-'*80}")
    print("STANDARD ALGORITHM")
    print(f"{'-'*80}")
    baseline = evaluate_multiplication(A, B, None, "Standard")
    print(f"Multiplications: {baseline.num_multiplications}")
    print(f"Time: {baseline.execution_time*1e6:.2f} µs")
    print(f"Correct: {baseline.is_correct}")
    print(f"Max error: {baseline.max_error:.2e}")
    
    # Test custom decompositions
    if decompositions:
        for name, decomp in decompositions.items():
            print(f"\n{'-'*80}")
            print(f"{name.upper()}")
            print(f"{'-'*80}")
            
            result = evaluate_multiplication(A, B, decomp, name)
            print(f"Multiplications: {result.num_multiplications}")
            print(f"Reduction: {baseline.num_multiplications - result.num_multiplications} "
                  f"({100*(1 - result.num_multiplications/baseline.num_multiplications):.1f}% fewer)")
            print(f"Time: {result.execution_time*1e6:.2f} µs")
            print(f"Speedup: {baseline.execution_time/result.execution_time:.2f}x")
            print(f"Correct: {result.is_correct}")
            print(f"Max error: {result.max_error:.2e}")
    
    print(f"\n{'='*80}\n")


def demo_2x2_real():
    """Demonstrate 2×2 real matrix multiplication."""
    print("\n" + "="*80)
    print("DEMO: 2×2 Real Matrix Multiplication")
    print("="*80)
    
    A = np.array([[1, 2],
                  [3, 4]], dtype=np.float64)
    
    B = np.array([[5, 6],
                  [7, 8]], dtype=np.float64)
    
    print("\nMatrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nExpected result (A @ B):")
    print(A @ B)
    
    # For this demo, we'll use standard algorithm
    # (Strassen's decomposition representation above is simplified)
    compare_methods(A, B)


def demo_3x3_random():
    """Demonstrate 3×3 random matrix multiplication."""
    print("\n" + "="*80)
    print("DEMO: 3×3 Random Matrix Multiplication")
    print("="*80)
    
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    
    print("\nMatrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nExpected result (A @ B):")
    print(A @ B)
    
    compare_methods(A, B)


def demo_4x4_complex():
    """Demonstrate 4×4 complex matrix multiplication."""
    print("\n" + "="*80)
    print("DEMO: 4×4 Complex Matrix Multiplication")
    print("="*80)
    print("(This is where AlphaEvolve achieved 48 multiplications vs Strassen's 49)")
    
    np.random.seed(42)
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    B = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    
    print("\nMatrix A (complex):")
    print(A)
    print("\nMatrix B (complex):")
    print(B)
    print("\nExpected result (A @ B):")
    print(A @ B)
    
    # AlphaEvolve's decomposition would go here
    decompositions = {}
    alphaevolve_decomp = alphaevolve_4x4_complex_decomposition()
    if alphaevolve_decomp:
        decompositions["AlphaEvolve (48 mults)"] = alphaevolve_decomp
    else:
        print("\nNOTE: AlphaEvolve's 48-multiplication decomposition not loaded.")
        print("This would require the actual triplets from their published results.")
    
    compare_methods(A, B, decompositions)


def verify_decomposition_correctness(m: int, n: int, p: int,
                                     decomposition: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                     num_tests: int = 100) -> Tuple[bool, float]:
    """
    Verify that a decomposition correctly implements matrix multiplication.
    
    Args:
        m, n, p: Matrix dimensions
        decomposition: The tensor decomposition to verify
        num_tests: Number of random test cases
        
    Returns:
        Tuple of (all_correct, max_error_observed)
    """
    max_error = 0.0
    all_correct = True
    
    for i in range(num_tests):
        # Generate random test matrices
        A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        B = np.random.randn(n, p) + 1j * np.random.randn(n, p)
        
        # Compute using decomposition
        multiplier = TensorDecompositionMultiplier(m, n, p, decomposition)
        result, _ = multiplier.multiply(A, B)
        
        # Compare to ground truth
        expected = A @ B
        error = np.max(np.abs(result - expected))
        max_error = max(max_error, error)
        
        if error > 1e-10:
            all_correct = False
    
    return all_correct, max_error


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║           AlphaEvolve Matrix Multiplication Evaluator                 ║
    ║                                                                        ║
    ║  Evaluates tensor decomposition-based matrix multiplication           ║
    ║  algorithms discovered by AlphaEvolve                                 ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demonstrations
    demo_2x2_real()
    demo_3x3_random()
    demo_4x4_complex()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
This evaluator provides:
✓ Correctness verification (comparing to standard NumPy multiplication)
✓ Multiplication count (the key metric - lower is better)
✓ Performance timing (though for small matrices, overhead dominates)
✓ Support for both real and complex matrices

To use AlphaEvolve's actual discoveries:
1. Load the decomposition triplets from their published results
2. Pass them to the TensorDecompositionMultiplier
3. The evaluator will count multiplications and verify correctness

The key insight: A rank-R decomposition needs exactly R scalar multiplications!
    """)