"""
MODEL.PY - Tensor Decomposition Optimizer (LLMGE will evolve this)

This contains the algorithm that SEARCHES for low-rank matrix multiplication algorithms.

AlphaEvolve evolved:
- Loss functions (reconstruction + discretization penalties)
- Optimizers (Adam → AdamW with weight decay)
- Initialization strategies  
- Noise injection for exploration
- Cyclical annealing schedules

LLMGE will modify the code between EVOLVE markers.
"""

import numpy as np


class TensorDecompositionOptimizer:
    """Searches for low-rank tensor decompositions of matrix multiplication."""
    
    def __init__(self, m: int, n: int, p: int, rank: int):
        self.m = m
        self.n = n  
        self.p = p
        self.rank = rank
        self.target = self._create_multiplication_tensor()
    
    def _create_multiplication_tensor(self):
        """Creates the 3D tensor representing matrix multiplication."""
        T = np.zeros((self.m * self.n, self.n * self.p, self.m * self.p), 
                     dtype=np.complex128)
        
        for i in range(self.m):
            for j in range(self.n):
                for k in range(self.p):
                    a_idx = i * self.n + j
                    b_idx = j * self.p + k
                    c_idx = i * self.p + k
                    T[a_idx, b_idx, c_idx] = 1.0
        
        return T
    
    # --EVOLVE--
    def optimize(self, num_steps=2000, learning_rate=0.01, seed=42):
        """Main optimization loop - LLMGE evolves this!"""
        np.random.seed(seed)
        
        # Initialize factors
        decomposition = []
        init_scale = 0.1
        for _ in range(self.rank):
            u = (np.random.randn(self.m * self.n) + 1j * np.random.randn(self.m * self.n)) * init_scale
            v = (np.random.randn(self.n * self.p) + 1j * np.random.randn(self.n * self.p)) * init_scale
            w = (np.random.randn(self.m * self.p) + 1j * np.random.randn(self.m * self.p)) * init_scale
            decomposition.append([u, v, w])
        
        # Simple gradient descent
        for step in range(num_steps):
            loss = self._compute_loss(decomposition)
            grads = self._compute_gradients(decomposition)
            
            for i in range(len(decomposition)):
                decomposition[i][0] -= learning_rate * grads[i][0]
                decomposition[i][1] -= learning_rate * grads[i][1]
                decomposition[i][2] -= learning_rate * grads[i][2]
            
            if step % 500 == 0:
                print(f"  Step {step:4d}, Loss: {loss:.6f}")
        
        return decomposition, loss
    
    def _compute_loss(self, decomposition):
        """Reconstruction loss."""
        reconstructed = np.zeros_like(self.target)
        for u, v, w in decomposition:
            reconstructed += np.einsum('i,j,k->ijk', u, v, w)
        diff = reconstructed - self.target
        return np.mean(np.abs(diff) ** 2)
    
    def _compute_gradients(self, decomposition, epsilon=1e-5):
        """Numerical gradients."""
        grads = []
        for idx, (u, v, w) in enumerate(decomposition):
            grad_triplet = []
            for factor in [u, v, w]:
                grad = np.zeros_like(factor)
                indices = np.random.choice(len(factor), min(10, len(factor)), replace=False)
                for i in indices:
                    original = factor[i]
                    factor[i] = original + epsilon
                    loss_plus = self._compute_loss(decomposition)
                    factor[i] = original - epsilon
                    loss_minus = self._compute_loss(decomposition)
                    factor[i] = original
                    grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
                grad_triplet.append(grad)
            grads.append(grad_triplet)
        return grads
    # --EVOLVE--
    
    def round_to_exact(self, decomposition):
        """Round to integer/half-integer values."""
        def round_half_int(x):
            return np.round(x * 2) / 2
        rounded = []
        for u, v, w in decomposition:
            rounded.append((round_half_int(np.array(u)), 
                          round_half_int(np.array(v)), 
                          round_half_int(np.array(w))))
        return rounded


class Model:
    """Main Model class that LLMGE expects."""
    
    def __init__(self, m=2, n=2, p=2, target_rank=7):
        self.m = m
        self.n = n
        self.p = p
        self.target_rank = target_rank
        self.decomposition = None
        self.optimizer = TensorDecompositionOptimizer(m, n, p, target_rank)
    
    def fit(self, num_steps=2000, learning_rate=0.01, seed=42):
        """Run optimization to find decomposition."""
        print(f"\nSearching for rank-{self.target_rank} decomposition "
              f"for {self.m}×{self.n} × {self.n}×{self.p} matrices...")
        
        decomposition, loss = self.optimizer.optimize(num_steps, learning_rate, seed)
        self.decomposition = self.optimizer.round_to_exact(decomposition)
        
        print(f"  Final loss: {loss:.6f}")
        return self
    
    def get_decomposition(self):
        """Return the found decomposition."""
        return self.decomposition


if __name__ == "__main__":
    print("="*80)
    print("Testing model.py")
    print("="*80)
    model = Model(m=2, n=2, p=2, target_rank=7)
    model.fit(num_steps=1000, learning_rate=0.01, seed=42)
    print(f"\nDecomposition rank: {len(model.get_decomposition())}")
    print("="*80)