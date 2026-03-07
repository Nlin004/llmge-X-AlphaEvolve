import argparse
import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
from pathlib import Path as p
from os.path import join as pj

# PROBLEM DEFINITION:
# This section defines the tensor decomposition problem to solve.
# If you are an LLM, consider evolving these parameters to discover better algorithms.
#
# CURRENT OBJECTIVE:
#   Find efficient decompositions for matrix multiplication
#
# PROBLEM TYPES:
#   - Square: N×N × N×N (e.g., 2×2 × 2×2, Strassen rank-7)
#   - Rectangular: N×M × M×P (e.g., 2×3 × 3×2, more general)
#
# EVOLUTION SUGGESTIONS FOR THE CONFIG:
#   - Try lower ranks to beat known benchmarks. DO NOT TOUCH THE DIMENSIONS (N,M,P), ONLY R.
#   - Modify loss function to bias toward integer factors
# =============================================================================

PROBLEM_CONFIG = {
    'N': 2,    # Rows of first matrix (A is N×M)
    'M': 2,    # Columns of A / Rows of B (shared dimension)
    'P': 2,    # Columns of second matrix (B is M×P, result is N×P)
    'R': 7,    # Target rank (scalar multiplications)
}

OPTIMIZATION_CONFIG = {
    'lr': 0.01,
    'iterations': 10000,
    'batch_size': 250,
}

# Known benchmarks (for square matrices):
KNOWN_SQUARE_RANKS = {
    (2, 2, 2): 7,   # Strassen (1969)
    (3, 3, 3): 23,  # AlphaEvolve (2022)
    (4, 4, 4): 49,
}

# --- 1. Utilities for Directory Management ---

def create_save_dir(save_root: str) -> str:
    """
    Create an incrementing exp directory (exp1, exp2, ...).
    """
    save_root_path = p(save_root)
    if not save_root_path.exists():
        save_root_path.mkdir(exist_ok=True, parents=True)

    n = []
    for exp_dir in save_root_path.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            # extract number from "expN"
            if exp_name.startswith("exp") and exp_name[3:].isdigit():
                try:
                    n.append(int(exp_name[3:]))
                except Exception:
                    pass

    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = pj(save_root, f"exp{sorted(n)[-1] + 1}")

    p(save_dir).mkdir(exist_ok=True, parents=True)
    return save_dir

# --- 2. Core Math / Logic ---

def generate_matmul_tensor(n: int) -> jnp.ndarray:
    """Generate the matrix multiplication tensor <n, n, n>."""
    dim = n * n
    T = jnp.zeros((dim, dim, dim))
    
    # Standard matrix multiplication tensor construction
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Flattens indices: (row * Width + col)
                T = T.at[i * n + j, j * n + k, k * n + i].set(1.0)
                # T = T.at[i * n + j, j * n + k, i * n + k].set(1.0) # Fixed: encode C[k,i] instead of C[i,k] -  old code computed transpose of the result.
    return T

def generate_matmul_tensor(n: int, m: int, p: int) -> jnp.ndarray: # THIS IS FOR RECNTAGULAR MATRIX MULTIPLICATION
    """
    Generate the matrix multiplication tensor for NM by MP -> NP.
    
    Args:
        n: Rows of first matrix A
        m: Columns of A / Rows of B (shared dimension)
        p: Columns of second matrix B
    
    Returns:
        Tensor of shape (n*m, m*p, n*p) encoding C[i,k] = sum_j A[i,j] * B[j,k]
    """
    T = jnp.zeros((n * m, m * p, n * p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Correct indexing for rectangular multiplication
                # A[i,j] is at position i*m+j in A_flat
                # B[j,k] is at position j*p+k in B_flat
                # C[i,k] is at position i*p+k in C_flat
                T = T.at[i * m + j, j * p + k, i * p + k].set(1.0)
    
    return T

# --- 3. Configuration ---

def get_args():
    parser = argparse.ArgumentParser()

    # Core problem config (use PROBLEM_CONFIG as defaults)
    parser.add_argument("--N", type=int, default=PROBLEM_CONFIG['N'], 
                       help="rows of first matrix (A is N x M)")
    parser.add_argument("--M", type=int, default=PROBLEM_CONFIG['M'], 
                       help="cols of A / rows of B (shared dimension)")
    parser.add_argument("--P", type=int, default=PROBLEM_CONFIG['P'], 
                       help="cols of second matrix (B is M x P, result is N x P)")
    parser.add_argument("--R", type=int, default=PROBLEM_CONFIG['R'], 
                       help="target rank")
    # parser.add_argument("--N", type=int, default=2, help="matrix size N (tensor dimension uses N*N)")
    # parser.add_argument("--R", type=int, default=7, help="rank for CP decomposition")

    # Optimization config

    parser.add_argument("--lr", type=float, default=OPTIMIZATION_CONFIG['lr'], help="learning rate")
    parser.add_argument("--iterations", type=int, default=OPTIMIZATION_CONFIG['iterations'], help="number of optimization steps")
    parser.add_argument("--batch_size", type=int, default=OPTIMIZATION_CONFIG['batch_size'], help="number of parallel random restarts")
    # parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    # parser.add_argument("--iterations", type=int, default=10000, help="number of optimization steps")
    # parser.add_argument("--batch_size", type=int, default=250, help="number of parallel random restarts")

    # Reproducibility / Output
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save_root", type=str, default="trained", help="root directory for experiment folders")
    parser.add_argument("--save_dir", type=str, default=None, help="Explicit experiment directory (used by eval.py)") ########
    parser.add_argument("--print_every", type=int, default=1000, help="print best loss every K iterations")

    return parser.parse_args()


def resolve_exp_dir(args):
    if args.save_dir is not None:
        exp_dir = p(args.save_dir).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    else:
        return p(create_save_dir(args.save_root)).resolve()
# --- 4. Main Execution ---
# --OPTION--

def main():
    args = get_args()
    exp_dir = resolve_exp_dir(args)
    thisFileName = os.path.basename(__file__)

    # Prob definition here:
    N = args.N  # Rows of A
    M = args.M  # Cols of A / Rows of B
    P = args.P  # Cols of B
    R = args.R  # Target rank

    # Derived values
    # N = args.N
    # dim = N * N
    # R = args.R

    # Tensor dimensions
    dim_A = N * M  # Flattened size of A
    dim_B = M * P  # Flattened size of B
    dim_C = N * P  # Flattened size of C (result)

    # Standard algorithm rank
    # standard_rank = N ** 3
    standard_rank = N * M * P 

    # Start with target rank, try to go lower
    if args.R is not None:
        start_rank = args.R
    else:
        start_rank = standard_rank - 1  # Start just below standard

    # Device info
    try:
        device_kind = jax.devices()[0].platform
    except:
        device_kind = "unknown"


    print(f"\n\n\n+================================[ Begin {thisFileName[:-3].upper()} Output ]================================+")
    print(f"    Running AlphaTensor-style Search")
    # print(f"    Matrix Size N: {N} (Tensor Dim: {dim})")
    print(f"    Problem: {N}×{M} × {M}×{P} => {N}×{P}")
    print(f"    Tensor dims: ({dim_A}, {dim_B}, {dim_C})")
    print(f"    Target Rank R: {R}")
    print(f"    Batch Size: {args.batch_size}")
    print(f"    Iterations: {args.iterations}")
    print(f"    Learning Rate: {args.lr}")
    print(f"    Device: {device_kind}")
    print("    "+"=" * 70)

    # 1. Target
    # target_tensor = generate_matmul_tensor(N)
    target_tensor = generate_matmul_tensor(N, M, P)

# --OPTION--
    # 2. Loss Function: 
    # IF YOU ARE AN LLM GIVEN A TASK TO EVOLVE THIS FUNCTION, SEE THE OTHER COMMENTED LOSS_FNs for ideas on how to modify it.
    # End goal is to provide a loss function that optimizes for solution factors as close as possible to a clean integer factorization (with factors ideally in -1,0,1) that reconstructs the tensor exactly.

    # LOSS FUNCTION WITHOUT DISCRETIZATION
    def loss_fn(params):
        U, V, W = params
        # Reconstruct: sum_r (u_r (x) v_r (x) w_r)
        # Factors shape: (dim, Rank)
        T_hat = jnp.einsum('ir,jr,kr->ijk', U, V, W)
        return jnp.sum((target_tensor - T_hat) ** 2)
    # =====================================================

    # THIS LOSS FUNCTION HAS DISCRETINIZATION PENALTY, WE WANT TO OPTIMIZE MORE FOR CORRECT WHEN ROUNDED!!!
    # def loss_fn(params, disc_weight):
    #     U, V, W = params
    #     T_hat = jnp.einsum('ir,jr,kr->ijk', U, V, W)
    #     reconstruction_loss = jnp.sum((target_tensor - T_hat) ** 2)
        
    #     # Always compute discretization penalty (but multiply by weight)
    #     # This avoids the if statement that causes JAX tracing issues
    #     def half_integer_distance(x):
    #         rounded = jnp.round(x * 2) / 2
    #         return jnp.sum((x - rounded) ** 2)
        
    #     disc_penalty = (half_integer_distance(U) + 
    #                 half_integer_distance(V) + 
    #                 half_integer_distance(W))
        
    #     # When disc_weight=0, this just adds 0
    #     return reconstruction_loss + disc_weight * disc_penalty
    # =========================================================

    # Loss function to bias towards Strassen-like solutions (with -1,0,1 coefficients):
    # def loss_fn(params):
    #     U, V, W = params
    #     T_hat = jnp.einsum('ir,jr,kr->ijk', U, V, W)

    #     # 1. Exact reconstruction pressure
    #     recon_loss = jnp.sum((target_tensor - T_hat) ** 2)

    #     # 2. Sparsity (L1)
    #     sparsity = (
    #         jnp.sum(jnp.abs(U)) +
    #         jnp.sum(jnp.abs(V)) +
    #         jnp.sum(jnp.abs(W))
    #     )

    #     # 3. Integer attraction (soft projection)
    #     def integer_distance(x):
    #         return jnp.sum((x - jnp.round(x)) ** 2)

    #     integer_penalty = (
    #         integer_distance(U) +
    #         integer_distance(V) +
    #         integer_distance(W)
    #     )

    #     return (
    #         recon_loss
    #         + 1e-4 * sparsity
    #         + 1e-3 * integer_penalty
    #     )
    # =========================================================
    # Strassen like loss function with annealed discretization penalty:
    # def loss_fn(params, disc_weight):  # Added disc_weight!
    #     U, V, W = params
    #     T_hat = jnp.einsum('ir,jr,kr->ijk', U, V, W)
    #     recon_loss = jnp.sum((target_tensor - T_hat) ** 2)
        
    #     # Half-integer penalty
    #     def half_integer_distance(x):
    #         rounded = jnp.round(x * 2) / 2
    #         return jnp.sum((x - rounded) ** 2)
        
    #     disc_penalty = (
    #         half_integer_distance(U) +
    #         half_integer_distance(V) +
    #         half_integer_distance(W)
    #     )
        
    #     return recon_loss + disc_weight * disc_penalty

# --OPTION--
    # If you are an LLM and you are tasked with modifying this optimizer, consider how you might implement a custom optimization loop that incorporates second-order information,
    #  adaptive learning rates, or even a custom update rule inspired by evolutionary strategies. 
    # You could also experiment with different optimizers available in Optax, such as RMSProp, Adagrad, or even more advanced ones like AdamW or LAMB. 
    # Additionally, you might want to implement a learning rate scheduler that adjusts the learning rate based on the training progress or validation performance.
    # ensure that your returns and parameters are compatible with potentially other pieces of the pipeline.

    # 3. Optimizer & Step
    optimizer = optax.adam(args.lr)
    @jax.jit
    # ============================WITHOUT ROUNDING====================================
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    # ============================WITH ROUNDING PENALTY====================================
    # def step(params, opt_state, disc_weight):
    #     loss, grads = jax.value_and_grad(lambda p: loss_fn(p, disc_weight))(params)
    #     updates, opt_state = optimizer.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)
    #     return params, opt_state, loss

        
# --OPTION--
    # If you are an LLM and you are tasked with modifying the below code: consider how you might implement a more sophisticated initialization strategy for the factor matrices U, V, W. 
    # For example, you could use a truncated SVD of the target tensor to initialize the factors in a way that captures more of the structure of the problem from the start. 
    # Alternatively, you could implement a custom random initialization that biases the factors towards certain patterns (e.g., sparsity, low magnitude, etc.) that you hypothesize might be beneficial for convergence.

    # 4. Initialization (Batched)
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.batch_size)

    def init_params(k):
        k1, k2, k3 = jax.random.split(k, 3)
        # return [
        #     jax.random.normal(k1, (dim, R)),
        #     jax.random.normal(k2, (dim, R)),
        #     jax.random.normal(k3, (dim, R))
        # ]
        return [
            jax.random.normal(k1, (dim_A, R)),
            jax.random.normal(k2, (dim_B, R)),
            jax.random.normal(k3, (dim_C, R))
        ]

    # Create batch of params and optimizer states
    batch_params = jax.vmap(init_params)(keys) 
    batch_opt_state = jax.vmap(optimizer.init)(batch_params)
# --OPTION--
    # If you are an LLM tasked with modifying the below code, consider 
    # 5. Training Loop
    # ============================WITHOUT ROUNDING====================================
    print("\n    Starting optimization...")
    
    for i in range(args.iterations):
        batch_params, batch_opt_state, batch_losses = jax.vmap(step)(batch_params, batch_opt_state)
        
        if i % args.print_every == 0:
            best_loss = jnp.min(batch_losses)
            print(f"       Step {i}: Best Loss = {best_loss:.6f}")

    # ============================WITH ROUNDING PENALTY====================================
    # # Training Loop with Annealing
    # print("\nStarting optimization...")
    # # MODIFY THESE TO CHANGE ANNEALING SCHEDULE
    # disc_weight_start = 0.0
    # disc_weight_end = 0.1
    # disc_anneal_start = 2000
    
    # for i in range(args.iterations):
    #     # Compute discretization weight (anneal from start to end)
    #     if i < disc_anneal_start:
    #         disc_weight = disc_weight_start
    #     else:
    #         progress = (i - disc_anneal_start) / (args.iterations - disc_anneal_start)
    #         disc_weight = disc_weight_start + progress * (disc_weight_end - disc_weight_start)
        
    #     # Vectorized step with current disc_weight
    #     batch_params, batch_opt_state, batch_losses = jax.vmap(
    #         lambda p, s: step(p, s, disc_weight)
    #     )(batch_params, batch_opt_state)
        
    #     if i % args.print_every == 0:
    #         best_loss = jnp.min(batch_losses)
    #         print(f"    Step {i}: Best Loss = {best_loss:.6f}, Disc Weight = {disc_weight:.4f}")
    # ====================================================================================

# --OPTION--

    # 6. Post-Processing
    min_loss_idx = jnp.argmin(batch_losses)
    best_params = jax.tree.map(lambda x: x[min_loss_idx], batch_params)
    final_loss = float(batch_losses[min_loss_idx])

    print("   "+"-" * 30)
    print(f"    Final Best Loss: {final_loss:.6f}")

    # Save results to file
    results_file = pj(exp_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Final Loss: {final_loss}\n")
        f.write(f"Problem: {N}x{M} x {M}x{P}\n") ############
        f.write(f"Config: {vars(args)}\n")
    print(f"    Results saved to {results_file}")
    
# --OPTION--

    # 7. Save Factors
    U, V, W = best_params
    
    # Convert to numpy for easier comparison logic
    factors_np = (np.array(U), np.array(V), np.array(W))

    # problem_data = (args.N, args.R)
    # np.savez(pj(exp_dir, "factors.npz"), U=factors_np[0], V=factors_np[1], W=factors_np[2], N=problem_data[0], R=problem_data[1])

    np.savez(pj(exp_dir, "factors.npz"), 
        U=factors_np[0], 
        V=factors_np[1], 
        W=factors_np[2], 
        N=N, M=M, P=P, R=R) # save problem stuff too so eval can unpack

    print(f"    Factors + problem config saved to {pj(exp_dir, 'factors.npz')}")

    print(f"+=================================[END OF {thisFileName[:-3].upper()} OUTPUT]=================================+\n") # name of file except the .py extension.
# --OPTION--

if __name__ == "__main__":
    # Ensure we run from the script location context if needed
    # script_directory = p(__file__).parent.resolve()
    # os.chdir(script_directory)
    main()