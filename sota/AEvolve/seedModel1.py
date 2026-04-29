# --OPTION--
# =============================================================================
# EVOLUTIONARY SEARCH CONFIGURATION
# =============================================================================
# This section defines the tensor decomposition problem to solve.
# If you are an LLM, consider evolving the parameters inside PROBLEM_CONFIG or OPTIMIZATION_CONFIG to discover better algorithms.
#
# CURRENT OBJECTIVE:
#   Find efficient decompositions for matrix multiplication
#
# PROBLEM TYPES:
#   - Square: N×N × N×N (e.g., 2×2 × 2×2, Strassen rank-7)
#   - Rectangular: N×M × M×P (e.g., 2×3 × 3×2, more general)
#
# EVOLUTION SUGGESTIONS FOR THE CONFIG:
#   - Modify the optimization config should you see fit (e.g., learning rate, batch size, iterations).
# =============================================================================

PROBLEM_CONFIG = {
    'N': 2,    # Rows of first matrix (A is N×M)
    'M': 2,    # Columns of A / Rows of B (shared dimension)
    'P': 2,    # Columns of second matrix (B is M×P, result is N×P)
    # NEVER EVER CHANGE THE ABOVE DIMENSIONS.
    'R': 7,    # Target rank (scalar multiplications). This is the target for optimization, but we can try lower or higher values. Changing this is NOT done in this code block, keep as is.
}

# Feel free to modify.
OPTIMIZATION_CONFIG = { 
    'lr': 0.01,
    'iterations': 10000,
    'batch_size': 250,
}

# Known benchmarks (for square matrices): THIS IS JUST FOR REFERENCE, THIS ISN'T CALLED ANYWHERE.
KNOWN_SQUARE_RANKS = {
    (2, 2, 2): 7,   # Strassen (1969)
    (3, 3, 3): 23,  # AlphaEvolve (2022)
    (4, 4, 4): 49,
}
# --OPTION--

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import os
import sys
import jax
import jax.numpy as jnp
import optax
from typing import List, Optional
import numpy as np
from pathlib import Path as p
from os.path import join as pj

# =============================================================================
# UTILITY FUNCTIONS (DO NOT MODIFY THESE!)
# =============================================================================

def create_save_dir(save_root: str) -> str:
    """Create an incrementing exp directory (exp1, exp2, ...)."""
    save_root_path = p(save_root)
    if not save_root_path.exists():
        save_root_path.mkdir(exist_ok=True, parents=True)

    n = []
    for exp_dir in save_root_path.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
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

def resolve_exp_dir(args):
    """Resolve experiment directory from args."""
    if args.save_dir is not None:
        exp_dir = p(args.save_dir).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    else:
        return p(create_save_dir(args.save_root)).resolve()

def create_sparkline_full_history(losses: List[float], width: int = 30):
    """Create sparkline showing FULL history from start to current."""
    if len(losses) < 2:
        return '▁' * width
    
    if len(losses) > width:
        indices = np.linspace(0, len(losses) - 1, width, dtype=int)
        display_values = [losses[i] for i in indices]
    else:
        display_values = losses.copy()
        padding = width - len(display_values)
        if padding > 0:
            display_values = [losses[0]] * padding + display_values
    
    vmin, vmax = min(display_values), max(display_values)
    
    if vmax == vmin:
        return '▄' * width
    
    if vmin > 0 and vmax / vmin > 1000:
        display_values = np.log10(display_values)
        vmin, vmax = min(display_values), max(display_values)
    
    sparks = ' ▁▂▃▄▅▆▇█'
    sparkline = []
    
    for val in display_values:
        normalized = (val - vmin) / (vmax - vmin)
        idx = int(normalized * 8)
        idx = max(0, min(8, idx))
        sparkline.append(sparks[idx])
    
    return ''.join(sparkline)

def generate_matmul_tensor(n: int, m: int, p: int) -> jnp.ndarray:
    """
    Generate the matrix multiplication tensor for N×M × M×P → N×P.
    
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
                T = T.at[i * m + j, j * p + k, i * p + k].set(1.0)
    
    return T

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--N", type=int, default=PROBLEM_CONFIG['N'], 
                       help="rows of first matrix (A is N x M)")
    parser.add_argument("--M", type=int, default=PROBLEM_CONFIG['M'], 
                       help="cols of A / rows of B (shared dimension)")
    parser.add_argument("--P", type=int, default=PROBLEM_CONFIG['P'], 
                       help="cols of second matrix (B is M x P, result is N x P)")
    parser.add_argument("--R", type=int, default=PROBLEM_CONFIG['R'], 
                       help="target rank")
    parser.add_argument("--lr", type=float, default=OPTIMIZATION_CONFIG['lr'], 
                       help="learning rate")
    parser.add_argument("--iterations", type=int, default=OPTIMIZATION_CONFIG['iterations'], 
                       help="number of optimization steps")
    parser.add_argument("--batch_size", type=int, default=OPTIMIZATION_CONFIG['batch_size'], 
                       help="number of parallel random restarts")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save_root", type=str, default="trained", 
                       help="root directory for experiment folders")
    parser.add_argument("--save_dir", type=str, default=None, 
                       help="Explicit experiment directory (used by eval.py)")
    parser.add_argument("--print_every", type=int, default=1000, 
                       help="print best loss every K iterations")
    
    return parser.parse_args()

# --OPTION--
# =============================================================================
# EVOLVABLE COMPONENTS (LLM CAN MODIFY THESE)
# =============================================================================

def create_loss_function(target_tensor):
    """
    Create the loss function for optimization. IT IS CRUCIAL THAT THIS FUNCTION 
    RETURNS A FUNCTION WITH THE SIGNATURE loss_fn(params) or loss_fn(params, disc_weight)
    depending on your implementation choice, as this is what the optimization loop 
    expects when it calls jax.value_and_grad(loss_fn)(params).
    
    LLM EVOLUTION GUIDE:
    You can modify this function to improve convergence and solution quality.
    
    CURRENT APPROACH:
    - Simple MSE reconstruction loss
    - No regularization or penalties
    
    EXAMPLE MODIFICATIONS (COME UP WITH YOUR OWN VARIATIONS TOO!):
    1. Add discretization penalty to encourage half-integer solutions:
       disc_penalty = sum((U - round(U*2)/2)^2) for U,V,W
       
    2. Add sparsity (L1) to encourage zeros:
       sparsity = |U| + |V| + |W|
       
    3. Add integer attraction:
       int_penalty = sum((U - round(U))^2) for U,V,W
       
    4. Combine multiple objectives:
       loss = reconstruction + λ1*discretization + λ2*sparsity
       
    5. Use annealed penalties (start low, increase over training)
    
    EXAMPLES:
    # Discretization only:
    def half_integer_distance(x):
        rounded = jnp.round(x * 2) / 2
        return jnp.sum((x - rounded) ** 2)
    disc = half_integer_distance(U) + half_integer_distance(V) + half_integer_distance(W)
    return reconstruction_loss + 0.01 * disc
    
    # Sparsity + Integer:
    sparsity = jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(V)) + jnp.sum(jnp.abs(W))
    int_penalty = jnp.sum((U - jnp.round(U))**2 + (V - jnp.round(V))**2 + (W - jnp.round(W))**2)
    return reconstruction_loss + 1e-4*sparsity + 1e-3*int_penalty
    """
    # IF YOU ARE AN LLM GIVEN A TASK TO EVOLVE THIS FUNCTION, SEE THE OTHER COMMENTED LOSS_FNs for ideas on how to modify it.
    # Your end goal is to provide a loss function that optimizes for solution factors as close as possible to a clean integer factorization (with factors ideally in -1,0,1) that reconstructs the tensor exactly.
    # Remember that T_hat in the existing loss function below is the reconstructed tensor from the given factors U, V, and W. 
    # The loss is currently just the L2 norm of the difference between the target tensor and the reconstructed tensor, which encourages exact reconstruction but does not directly encourage integer or sparse factors.
    
    # this INTERNAL loss function is called in the step function during optimization, in a line that'll look like this:
    #   loss, grads = jax.value_and_grad(loss_fn)(params)
    # Ensure that whatever improved version of loss_fn you create is compatible with a call with this usage.

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
    # def loss_fn(params):  # Added disc_weight!
    #     disc_weight = 0.01  # hard coded to allow SAME function signature for usage with other modified code blocks.
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
    
    return loss_fn

# --OPTION--


def create_optimizer(learning_rate):
    """
    Create the optimizer for training.
    
    LLM EVOLUTION GUIDE:
    You can modify this to use different optimizers or learning rate schedules.
    
    CURRENT APPROACH:
    - Adam with fixed learning rate
    
    EXAMPLE MODIFICATIONS (COME UP WITH YOUR OWN VARIATIONS TOO!):
    1. Different optimizers:
       - optax.sgd(learning_rate, momentum=0.9, nesterov=True)
       - optax.rmsprop(learning_rate)
       - optax.adamw(learning_rate, weight_decay=1e-4)
       - optax.lion(learning_rate)
       
    2. Learning rate schedules:
       schedule = optax.exponential_decay(
           init_value=0.01,
           transition_steps=1000,
           decay_rate=0.96
       )
       return optax.adam(schedule)
       
    3. Cosine annealing:
       schedule = optax.cosine_decay_schedule(
           init_value=0.01,
           decay_steps=10000,
           alpha=0.0
       )
       return optax.adam(schedule)
       
    4. Warmup + decay:
       schedule = optax.warmup_cosine_decay_schedule(
           init_value=0.0,
           peak_value=0.01,
           warmup_steps=1000,
           decay_steps=10000
       )
       return optax.adam(schedule)
       
    5. Adaptive gradient clipping:
       return optax.chain(
           optax.clip_by_global_norm(1.0),
           optax.adam(learning_rate)
       )
    """
    return optax.adam(learning_rate)




# --OPTION--
# functionality wise, this code block should not really be modified. all variable names are CRUCIAL to stay. We still want to call main through def main.
# =============================================================================
# MAIN EXECUTION (PRESERVE ORIGINAL LEVEL OF INDENTATION, relative to if __name__ == "__main__":)
# =============================================================================


# PRESERVE ANY ORIGINAL LEVELS OF INDENTATION BELOW!
def main():

    # Parse arguments and setup
    args = get_args()
    exp_dir = resolve_exp_dir(args)
    thisFileName = os.path.basename(__file__)

    print(jax.devices())
    jax.config.update("jax_platform_name", "gpu") 

    # Problem dimensions (DO NOT MODIFY N, M, P - only R can be evolved)
    N = args.N
    M = args.M
    P = args.P
    R = args.R  # LLM: You can try R-1, R+1, etc. to find new ranks

    # Derived values
    dim_A = N * M
    dim_B = M * P
    dim_C = N * P
    standard_rank = N * M * P

    # Device info
    try:
        device_kind = jax.devices()[0].platform
    except:
        device_kind = "unknown"

    # Print configuration
    print(f"\n\n\n+====================================[ Begin {thisFileName[:-3].upper()} Output ]====================================+")
    print(f"    Running AlphaTensor-style Search")
    print(f"    Problem: {N}×{M} × {M}×{P} => {N}×{P}")
    print(f"    Tensor Representation: <{N}, {M}, {P}>")
    print(f"    Target Rank R: {R}")
    print(f"    Batch Size: {args.batch_size}")
    print(f"    Iterations: {args.iterations}")
    print(f"    Learning Rate: {args.lr}")
    print(f"    Device: {device_kind}")
    print("    " + "=" * 90)

    # Generate target tensor (DO NOT MODIFY)
    target_tensor = generate_matmul_tensor(N, M, P)

    # Creates loss function 
    loss_fn = create_loss_function(target_tensor = target_tensor) # helper function call! be wary of modifying arguments here, unless you understand the function's behavior!
    # create_loss_function returns: jnp.sum((target_tensor - T_hat) ** 2)

    # Create optimizer
    optimizer = create_optimizer(learning_rate = args.lr) # helper function call! be wary of modifying, unless you understand the below:
    # the return for this function by default is: return optax.adam(learning_rate)


# --OPTION--
# PRESERVE ANY ORIGINAL LEVELS OF INDENTATION BELOW!
    @jax.jit # KEEP THIS! also ensure your step function works with jax.
    # Optimizer & Step

    # If you are an LLM and you are tasked with modifying this optimizer, consider how you might implement a custom optimization loop that incorporates second-order information,
    #  adaptive learning rates, or even a custom update rule inspired by evolutionary strategies. 
    # You could also experiment with different optimizers available in Optax, such as RMSProp, Adagrad, or even more advanced ones like AdamW or LAMB. 
    # Additionally, you might want to implement a learning rate scheduler that adjusts the learning rate based on the training progress or validation performance.
    # ensure that your returns and parameters are compatible with potentially other pieces of the pipeline.

    # Preserve original level of indentation! def step() below is indented one tab in (from def main). content inside def step is indented one tab further.
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
# PRESERVE ALL INDENTATION BELOW! It is part of the def main() workflow.
    # Initialize parameters (LLM can modify initialize_factors above)
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.batch_size)

    # Create initialization function
    def init_params(k):
        """
        Initialize factor matrices U, V, W.
        
        LLM EVOLUTION GUIDE:
        You can modify this to use smarter initialization strategies.
        
        CURRENT APPROACH:
        - Random normal initialization
        
        EXAMPLE MODIFICATIONS (COME UP WITH YOUR OWN VARIATIONS TOO!):
        1. Smaller initial values (better for discrete solutions):
        scale = 0.1
        return [
            jax.random.normal(k1, (dim_A, R)) * scale,
            jax.random.normal(k2, (dim_B, R)) * scale,
            jax.random.normal(k3, (dim_C, R)) * scale
        ]
        
        2. Uniform initialization in [-1, 1]:
        return [
            jax.random.uniform(k1, (dim_A, R), minval=-1.0, maxval=1.0),
            jax.random.uniform(k2, (dim_B, R), minval=-1.0, maxval=1.0),
            jax.random.uniform(k3, (dim_C, R), minval=-1.0, maxval=1.0)
        ]
        
        3. Initialize near discrete values {-1, 0, 1}:
        U_discrete = jax.random.choice(k1, jnp.array([-1., 0., 1.]), shape=(dim_A, R))
        U = U_discrete + jax.random.normal(k1, (dim_A, R)) * 0.01
        
        4. Strassen-inspired (for 2×2):
        if dim_A == 4 and R == 7:
            # Initialize with Strassen structure + noise
            U_base = jnp.array([[1,0,1,0,1,-1,0], [0,0,0,0,1,0,1], ...])
            U = U_base + jax.random.normal(k1, (4, 7)) * 0.1
            
        5. Xavier/He initialization:
        scale = jnp.sqrt(2.0 / (dim_A + R))
        return [
            jax.random.normal(k1, (dim_A, R)) * scale,
            ...
        ]
        """
        k1, k2, k3 = jax.random.split(k, 3)
        return [
            jax.random.normal(k1, (dim_A, R)),
            jax.random.normal(k2, (dim_B, R)),
            jax.random.normal(k3, (dim_C, R))
        ]

    # Initialize batch
    batch_params = jax.vmap(init_params)(keys)
    batch_opt_state = jax.vmap(optimizer.init)(batch_params)

# --OPTION--
# PRESERVE ALL ORIGINAL LEVELS OF INDENTATION RELATIVE TO THIS COMMENT BELOW!
    # assume previous code blocks gave us: 
        # batch_params = jax.vmap(init_params)(keys)
        # batch_opt_state = jax.vmap(optimizer.init)(batch_params)

    # Training loop
    # LLM EVOLUTION GUIDE:
    # You can modify the training loop to:
    # 1. Add early stopping when loss < threshold
    # 2. Implement progressive rank reduction (start high, reduce when converged)
    # 3. Add periodic projection to discrete values
    # 4. Implement simulated annealing (temperature-based search)
    # 5. Add basin hopping or other global optimization techniques
    # 6. Implement ensemble methods (train multiple models, combine)

    loss_history = []
    bar_width = 30
    spark_width = 30

    print("\n    Starting optimization...\n")

    for i in range(args.iterations):
        batch_params, batch_opt_state, batch_losses = jax.vmap(step)(batch_params, batch_opt_state)
        best_loss = float(jnp.min(batch_losses))
        loss_history.append(best_loss)
        
        # Update progress bar every 10 steps
        if i % 10 == 0 or i == args.iterations - 1:
            progress = (i + 1) / args.iterations
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            spark = create_sparkline_full_history(loss_history, width=spark_width) # helper function call.
            
            sys.stdout.write(
                f'\r    [{bar}] {progress:>5.1%} | '
                f'Step {i:5d}/{args.iterations} | '
                f'Loss: {best_loss:.6f} | '
                f'{spark}'
            )
            sys.stdout.flush()
        
        # Print checkpoint
        if i % args.print_every == 0:
            sys.stdout.write('\r' + ' ' * 150 + '\r')
            print(f"       Step {i}: Best Loss = {best_loss:.6f}")
            
            if i < args.iterations - 1:
                progress = (i + 1) / args.iterations
                filled = int(bar_width * progress)
                bar = '█' * filled + '░' * (bar_width - filled)
                spark = create_sparkline_full_history(loss_history, width=spark_width)
                sys.stdout.write(
                    f'\r    [{bar}] {progress:>5.1%} | '
                    f'Step {i:5d}/{args.iterations} | '
                    f'Loss: {best_loss:.6f} | '
                    f'{spark}'
                )
                sys.stdout.flush()
# --OPTION--
# PRESERVE ANY LEVELS OF INDENTATION IN YOUR CODE TO MATCH BELOW!

    # Extract best result and save
    min_loss_idx = jnp.argmin(batch_losses)
    best_params = jax.tree.map(lambda x: x[min_loss_idx], batch_params)
    final_loss = float(batch_losses[min_loss_idx])

    print(f"\n\n    Final Best Loss: {final_loss:.6f}")

    # Save results
    results_file = pj(exp_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Final Loss: {final_loss}\n")
        f.write(f"Problem: {N}x{M} x {M}x{P}\n")
        f.write(f"Config: {vars(args)}\n")
    print(f"    Results saved to {results_file}")

    # Save factors
    U, V, W = best_params
    factors_np = (np.array(U), np.array(V), np.array(W))

    np.savez(pj(exp_dir, "factors.npz"), 
        U=factors_np[0], 
        V=factors_np[1], 
        W=factors_np[2], 
        N=N, M=M, P=P, R=R)

    print(f"    Factors + problem config saved to {pj(exp_dir, 'factors.npz')}")
    print(f"+=================================[END OF {thisFileName[:-3].upper()} OUTPUT]=================================+\n")
# PRESERVE THE CODE ABOVE'S ORIGINAL LEVEL OF INDENTATION WHEN YOU MODIFY IT! NOTICE THIS IS ONE TAB IN FROM THE "if __name__ == '__main__':" GUARD. 
# DO NOT ADD ANOTHER LAYER OF INDENTATION UNLESS YOU ARE ADDING A NEW FUNCTION OR CLASS DEFINITION, ETC. THAT REQUIRES IT.




if __name__ == "__main__":
    main()