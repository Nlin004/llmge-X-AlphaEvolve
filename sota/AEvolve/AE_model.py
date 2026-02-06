import numpy as np


class TensorDecompositionOptimizer:
    """
    Searches for low-rank decompositions of the matrix multiplication tensor.

    IMPORTANT:
    - Loss computation is isolated
    - Gradients are numerical but non-recursive
    - This mirrors AlphaEvolve structure, minus autodiff
    """

    def __init__(self, m, n, p, rank):
        self.m = m
        self.n = n
        self.p = p
        self.rank = rank
        self.target = self._create_target_tensor()

    def _create_target_tensor(self):
        T = np.zeros((self.m * self.n,
                      self.n * self.p,
                      self.m * self.p))
        for i in range(self.m):
            for j in range(self.n):
                for k in range(self.p):
                    a = i * self.n + j
                    b = j * self.p + k
                    c = i * self.p + k
                    T[a, b, c] = 1.0
        return T

    def _compute_loss(self, factors):
        """
        Reconstruction loss only.
        No gradients. No recursion.
        """
        recon = np.zeros_like(self.target)
        for u, v, w in factors:
            recon += np.einsum("i,j,k->ijk", u, v, w)
        diff = recon - self.target
        return np.mean(diff ** 2)

    def _compute_gradients(self, factors, eps=1e-6, samples=8):
        """
        Finite-difference gradients.
        This is slow but correct and non-recursive.
        """
        grads = []

        base_loss = self._compute_loss(factors)

        for r, (u, v, w) in enumerate(factors):
            grad_u = np.zeros_like(u)
            grad_v = np.zeros_like(v)
            grad_w = np.zeros_like(w)

            for vec, grad in [(u, grad_u), (v, grad_v), (w, grad_w)]:
                idxs = np.random.choice(len(vec), min(samples, len(vec)), replace=False)
                for i in idxs:
                    old = vec[i]

                    vec[i] = old + eps
                    lp = self._compute_loss(factors)

                    vec[i] = old - eps
                    lm = self._compute_loss(factors)

                    vec[i] = old
                    grad[i] = (lp - lm) / (2 * eps)

            grads.append((grad_u, grad_v, grad_w))

        return grads

    def optimize(self, num_steps=2000, lr=1e-2, seed=0):
        np.random.seed(seed)

        factors = []
        scale = 0.1
        for _ in range(self.rank):
            u = np.random.randn(self.m * self.n) * scale
            v = np.random.randn(self.n * self.p) * scale
            w = np.random.randn(self.m * self.p) * scale
            factors.append([u, v, w])

        for step in range(num_steps):
            loss = self._compute_loss(factors)
            grads = self._compute_gradients(factors)

            for i in range(self.rank):
                factors[i][0] -= lr * grads[i][0]
                factors[i][1] -= lr * grads[i][1]
                factors[i][2] -= lr * grads[i][2]

            if step % 500 == 0:
                print(f"Step {step:4d} | loss = {loss:.6f}")

        return factors

    def round_factors(self, factors):
        """
        Project continuous factors onto a discrete set.

        This mimics AlphaEvolve's projection step from
        continuous search space to exact arithmetic.
        """
        def round_half_integers(x):
            return np.round(x * 2) / 2

        rounded = []
        for u, v, w in factors:
            rounded.append((
                round_half_integers(u.copy()),
                round_half_integers(v.copy()),
                round_half_integers(w.copy())
            ))
        return rounded


class Model:
    """
    Wrapper class expected by LLM-GE.
    """

    def __init__(self, m=2, n=2, p=2, target_rank=7):
        self.optimizer = TensorDecompositionOptimizer(m, n, p, target_rank)
        self.decomposition = None

    def fit(self, **kwargs):
        factors = self.optimizer.optimize(**kwargs)
        self.decomposition = self.optimizer.round_factors(factors)
        return self

    def get_decomposition(self):
        return self.decomposition