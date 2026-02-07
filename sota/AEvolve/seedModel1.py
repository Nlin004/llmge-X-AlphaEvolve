import jax
import jax.numpy as jnp
import optax

# 1. Configuration
N = 2
dim = N * N
R = 7
learning_rate = 0.01
iterations = 5000

# 2. Generate Ground Truth (Same logic, JAX syntax)
def generate_matmul_tensor(n):
    dim = n * n
    # Create grid of indices
    i, j, k = jnp.meshgrid(jnp.arange(n), jnp.arange(n), jnp.arange(n), indexing='ij')
    
    # Flatten indices to compute 1D positions
    idx_a = (i * n + k).flatten()
    idx_b = (k * n + j).flatten()
    idx_c = (i * n + j).flatten()
    
    # Create the tensor
    T = jnp.zeros((dim, dim, dim))
    # In JAX, we use .at[].set() for updates because arrays are immutable
    T = T.at[idx_a, idx_b, idx_c].set(1.0)
    return T

target_tensor = generate_matmul_tensor(N)

# 3. Define the Loss Function (Functional Style)
def loss_fn(params):
    U, V, W = params
    # Reconstruct: sum_r (u_r (x) v_r (x) w_r)
    T_hat = jnp.einsum('ir,jr,kr->ijk', U, V, W)
    # Frobenius Norm
    return jnp.sum((target_tensor - T_hat) ** 2)

# 4. The Optimization Step
optimizer = optax.adam(learning_rate)

@jax.jit
def step(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 5. Initialization (The JAX Way)
key = jax.random.PRNGKey(42)

# --- THE BIG BENEFIT: BATCHED INITIALIZATION ---
# Instead of 1 try, we define 100 tries at once
batch_size = 100 
keys = jax.random.split(key, batch_size)

def init_params(k):
    k1, k2, k3 = jax.random.split(k, 3)
    return [
        jax.random.normal(k1, (dim, R)),
        jax.random.normal(k2, (dim, R)),
        jax.random.normal(k3, (dim, R))
    ]

# Create 100 sets of parameters
batch_params = jax.vmap(init_params)(keys) 
# Create 100 optimizer states
batch_opt_state = jax.vmap(optimizer.init)(batch_params)

# 6. Training Loop (Running 100 optimizations in parallel)
print(f"Running {batch_size} parallel searches for Rank {R}...")

for i in range(iterations):
    # vmap the step function to run it on the whole batch
    batch_params, batch_opt_state, batch_losses = jax.vmap(step)(batch_params, batch_opt_state)
    
    if i % 1000 == 0:
        # Check the best loss across all 100 runs
        best_loss = jnp.min(batch_losses)
        print(f"Step {i}: Best Loss among batch = {best_loss:.6f}")

# 7. Check Results
min_loss_idx = jnp.argmin(batch_losses)
best_params = jax.tree_map(lambda x: x[min_loss_idx], batch_params)
final_loss = batch_losses[min_loss_idx]

print("-" * 30)
print(f"Best Final Loss: {final_loss:.6f}")

if final_loss < 1e-4:
    print("SUCCESS: Found a decomposition!")
    U, V, W = best_params
    print(jnp.round(U))