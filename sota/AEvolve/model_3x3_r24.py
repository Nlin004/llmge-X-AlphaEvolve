# --OPTION--
# =============================================================================
# EVOLUTIONARY SEARCH CONFIGURATION
# =============================================================================
# This section defines the tensor decomposition problem to solve.
# If you are an LLM, consider evolving the parameters inside OPTIMIZATION_CONFIG to discover better algorithms.
#
# CURRENT OBJECTIVE:
#   Find efficient decompositions for matrix multiplication
#
# NOTE: N, M, P, and R are overridden at runtime by eval.py via CLI arguments
# during dynamic rank search. The values below are only used as defaults when
# running this script standalone (outside of the LLMGE eval pipeline).
#
# EVOLUTION SUGGESTIONS:
#   - Modify OPTIMIZATION_CONFIG (learning rate, batch size, iterations).
# =============================================================================

# Feel free to modify.
OPTIMIZATION_CONFIG = {
    'lr': 0.01,
    'iterations': 10000,  # note: eval.py passes --iterations 3000 during rank search
    'batch_size': 250,
}

# Known benchmarks (for reference only — not called anywhere):
# KNOWN_SQUARE_RANKS = {
#     (2, 2, 2): 7,   # Strassen (1969)
#     (3, 3, 3): 23,  # AlphaEvolve (2022)
#     (4, 4, 4): 49,
# }
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
    
    parser.add_argument("--N", type=int, default=2, 
                       help="rows of first matrix (A is N x M)")
    parser.add_argument("--M", type=int, default=2, 
                       help="cols of A / rows of B (shared dimension)")
    parser.add_argument("--P", type=int, default=2, 
                       help="cols of second matrix (B is M x P, result is N x P)")
    parser.add_argument("--R", type=int, default=7, 
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
    parser.add_argument("--early_stop_threshold", type=float, default=0.0,
                       help="stop early if best loss falls below this value (0 = disabled)")

    return parser.parse_args()

# --OPTION--
def create_loss_function(target_tensor):
    """
    Create the loss function for optimization.
    
    Parameters:
        - target_tensor (jax array): Target tensor to be reconstructed.
    
    Returns:
        - A function with the signature loss_fn(params) that computes the loss.
    """

    # Define the loss function with integer attraction and sparsity
    def loss_fn(params):
        U, V, W = params
        # Reconstruct: sum_r (u_r (x) v_r (x) w_r)
        # Factors shape: (dim, Rank)
        T_hat = jnp.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Reconstruction loss
        reconstruction_loss = jnp.sum((target_tensor - T_hat) ** 2)
        
        # Sparsity (L1)
        sparsity = (
            jnp.sum(jnp.abs(U)) +
            jnp.sum(jnp.abs(V)) +
            jnp.sum(jnp.abs(W))
        )
        
        # Integer attraction (soft projection)
        def integer_distance(x):
            return jnp.sum((x - jnp.round(x)) ** 2)

        integer_penalty = (
            integer_distance(U) +
            integer_distance(V) +
            integer_distance(W)
        )
        
        # Combine losses with weights
        return (
            reconstruction_loss
            + 1e-4 * sparsity
            + 1e-3 * integer_penalty
        )
    
    return loss_fn

# --OPTION--

# Modify this optimizer as you see fit! You can also implement multiple optimizers and select between them via a command-line argument.
# BE SURE TO PRESERVE ALL LEVELS OF INDENTATION RELATIVE TO THIS COMMENT!
import optax

def create_optimizer(learning_rate=0.01, 
                     optimizer_type='nadam', 
                     clip_gradient=True, 
                     warmup_steps=1000, 
                     decay_steps=5000, 
                     peak_value=0.1):
    """
    Create the optimizer for training.
    
    Args:
        learning_rate (float): The initial learning rate.
        optimizer_type (str): Type of optimizer to use. Defaults to 'nadam'.
        clip_gradient (bool): Whether to clip gradients. Defaults to True.
        warmup_steps (int): Number of steps for warmup. Defaults to 1000.
        decay_steps (int): Number of steps for decay. Defaults to 5000.
        peak_value (float): Peak value for warmup cosine decay schedule. Defaults to 0.1.
    
    Returns:
        optax.GradientTransformation: The created optimizer.
    """

    # Simplify the learning rate schedule definition
    if warmup_steps > 0 and decay_steps > 0 and peak_value is not None:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps
        )
    elif decay_steps > 0:
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=decay_steps,
            alpha=0.0
        )
    else:
        schedule = learning_rate

    # Reduce the number of optimizers and simplify their definitions
    if optimizer_type =='sgd':
        optimizer = optax.sgd(schedule, momentum=0.95, nesterov=True)
    elif optimizer_type == 'adam':
        optimizer = optax.adam(schedule, b1=0.85, b2=0.99, eps=1e-8)
    elif optimizer_type == 'nadam':
        optimizer = optax.nadam(schedule, b1=0.85, b2=0.99, eps=1e-8)
    else:
        optimizer = optax.nadam(schedule, b1=0.85, b2=0.99, eps=1e-8)  # Default to Nadam

    # Remove unnecessary clipping parameters
    if clip_gradient:
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optimizer
        )

    return optimizer

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
        - Hybrid initialization combining uniform and normal distributions
        
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
            U_base = jnp.array([[1,0,1,0,1,-1,0], [0,0,0,0,1,0,1], [0,1,0,1,0,0,1], [0,0,1,0,0,1,0]])
            U = U_base + jax.random.normal(k1, (4, 7)) * 0.1
            
        5. Xavier/He initialization:
        scale = jnp.sqrt(2.0 / (dim_A + R))
        return [
            jax.random.normal(k1, (dim_A, R)) * scale,
            jax.random.normal(k2, (dim_B, R)) * scale,
            jax.random.normal(k3, (dim_C, R)) * scale
        ]
        """
        k1, k2, k3 = jax.random.split(k, 3)
        scale_uniform = 0.5
        scale_normal = 0.1
        return [
            jax.random.uniform(k1, (dim_A, R), minval=-scale_uniform, maxval=scale_uniform) + jax.random.normal(k1, (dim_A, R)) * scale_normal,
            jax.random.uniform(k2, (dim_B, R), minval=-scale_uniform, maxval=scale_uniform) + jax.random.normal(k2, (dim_B, R)) * scale_normal,
            jax.random.uniform(k3, (dim_C, R), minval=-scale_uniform, maxval=scale_uniform) + jax.random.normal(k3, (dim_C, R)) * scale_normal
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

        if args.early_stop_threshold > 0 and best_loss < args.early_stop_threshold:
            print(f"\n    Early stop at step {i}: loss {best_loss:.6f} < threshold {args.early_stop_threshold}")
            break

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