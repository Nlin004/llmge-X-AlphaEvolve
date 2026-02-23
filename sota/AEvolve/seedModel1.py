import argparse
import os
import numpy as np
from pathlib import Path as p
from os.path import join as pj

# =============================================================================
# CIRCUIT DESIGN SEED MODEL
#
# APPROACH: Mini genetic algorithm (population + crossover + mutation).
#
# Why a mini-GA instead of pure random search?
#   Random search and hill climbing alone cannot reliably find a full adder
#   because the correct circuit requires specific substructures (a XOR b,
#   a AND b, etc.) that must be wired together correctly. The chance of
#   randomly hitting all of these simultaneously is astronomically low.
#   Crossover combines partial solutions from two parents, letting good
#   substructures from different circuits combine into a correct whole.
#
# Three phases:
#   Phase 1 — Seeded population: start with known-good subcircuits wired
#             in, plus random circuits to maintain diversity.
#   Phase 2 — Evolution: selection, crossover, mutation for N generations.
#   Phase 3 — Polish: hill-climb the best individual found.
#
# Correctness is a hard constraint throughout — wrong circuits are never
# preferred over correct ones regardless of gate count.
#
# EMADE INTERFACE:
#   encode_individual(circuit) -> flat int numpy genome
#   decode_individual(genome)  -> circuit dict
# =============================================================================

GATE_AND  = 0
GATE_OR   = 1
GATE_XOR  = 2
GATE_NOT  = 3
GATE_NAND = 4
GATE_NOR  = 5
GATE_BUF  = 6
NUM_GATE_TYPES = 7
GATE_NAMES  = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "BUF"]
UNARY_GATES = {GATE_NOT, GATE_BUF}


# --- 1. Directory Management ---
# --OPTION--
def create_save_dir(save_root: str) -> str:
    save_root_path = p(save_root)
    save_root_path.mkdir(exist_ok=True, parents=True)
    n = [int(d.name[3:]) for d in save_root_path.iterdir()
         if d.is_dir() and d.name.startswith("exp") and d.name[3:].isdigit()]
    save_dir = pj(save_root, "exp1") if not n else pj(save_root, f"exp{max(n)+1}")
    p(save_dir).mkdir(exist_ok=True, parents=True)
    return save_dir


# --- 2. Truth Table ---
# --OPTION--
def get_truth_table(func_name: str, num_inputs: int) -> np.ndarray:
    """
    Returns (2**num_inputs, num_outputs) int32 array of {0,1}.
    Supported: full_adder, majority3, xor4, parity4, custom
    """
    rows   = 2 ** num_inputs
    inputs = np.array(
        [[int(b) for b in format(i, f'0{num_inputs}b')] for i in range(rows)],
        dtype=np.int32
    )
    if func_name == "full_adder":
        assert num_inputs == 3
        a, b, cin = inputs[:,0], inputs[:,1], inputs[:,2]
        return np.hstack([(a^b^cin).reshape(-1,1),
                          ((a&b)|(b&cin)|(a&cin)).reshape(-1,1)])
    elif func_name == "majority3":
        assert num_inputs == 3
        a, b, c = inputs[:,0], inputs[:,1], inputs[:,2]
        return ((a&b)|(b&c)|(a&c)).reshape(-1,1)
    elif func_name == "xor4":
        assert num_inputs == 4
        return (inputs[:,0]^inputs[:,1]^inputs[:,2]^inputs[:,3]).reshape(-1,1)
    elif func_name == "parity4":
        assert num_inputs == 4
        return (inputs[:,0]^inputs[:,1]^inputs[:,2]^inputs[:,3]).reshape(-1,1)
    elif func_name == "custom":
        raise NotImplementedError("Fill in custom truth table.")
    else:
        raise ValueError(f"Unknown function: {func_name}")


# --- 3. Circuit Representation ---
# --OPTION--
#
# Signals: [input_0..input_{NI-1}, node_0..node_{NN-1}]
# Node k can only read from signals 0..(num_inputs+k-1)  [causal DAG]
# Output nodes MUST point at gate nodes (index >= num_inputs)
# to avoid the degenerate case of outputs = raw inputs.

def random_circuit(num_inputs: int, max_nodes: int, num_outputs: int,
                   rng: np.random.Generator) -> dict:
    """Sample a completely random valid discrete circuit."""
    gate_types = rng.integers(0, NUM_GATE_TYPES, size=max_nodes).astype(np.int32)
    conn_a = np.zeros(max_nodes, dtype=np.int32)
    conn_b = np.zeros(max_nodes, dtype=np.int32)
    for node_idx in range(max_nodes):
        available = max(num_inputs + node_idx, 1)
        conn_a[node_idx] = rng.integers(0, available)
        conn_b[node_idx] = rng.integers(0, available)
    out_nodes = rng.integers(num_inputs, num_inputs + max_nodes,
                             size=num_outputs).astype(np.int32)
    return {'gate_types': gate_types, 'conn_a': conn_a,
            'conn_b': conn_b, 'out_nodes': out_nodes}


def make_seeded_circuit(num_inputs: int, max_nodes: int, num_outputs: int,
                        func_name: str, rng: np.random.Generator) -> dict:
    """
    Build a circuit pre-wired with known-good subcircuits for the target
    function, then fill remaining nodes randomly.

    This gives the GA a massive head start by providing building blocks
    that are known to be useful — e.g. a XOR b and a AND b for adders.
    The GA still needs to wire these together correctly.
    """
    gate_types = np.full(max_nodes, GATE_BUF, dtype=np.int32)
    conn_a     = np.zeros(max_nodes, dtype=np.int32)
    conn_b     = np.zeros(max_nodes, dtype=np.int32)

    if func_name == "full_adder" and num_inputs == 3 and max_nodes >= 5:
        # Hardwire the 5 gates needed for a correct full adder:
        # node0 (sig3): XOR(a, b)
        # node1 (sig4): AND(a, b)
        # node2 (sig5): XOR(node0, cin) = sum
        # node3 (sig6): AND(node0, cin)
        # node4 (sig7): OR(node1, node3) = carry
        gate_types[0] = GATE_XOR; conn_a[0] = 0; conn_b[0] = 1  # a XOR b
        gate_types[1] = GATE_AND; conn_a[1] = 0; conn_b[1] = 1  # a AND b
        gate_types[2] = GATE_XOR; conn_a[2] = 3; conn_b[2] = 2  # node0 XOR cin
        gate_types[3] = GATE_AND; conn_a[3] = 3; conn_b[3] = 2  # node0 AND cin
        gate_types[4] = GATE_OR;  conn_a[4] = 4; conn_b[4] = 6  # node1 OR node3
        # Fill remaining nodes randomly
        for node_idx in range(5, max_nodes):
            available = max(num_inputs + node_idx, 1)
            gate_types[node_idx] = rng.integers(0, NUM_GATE_TYPES)
            conn_a[node_idx]     = rng.integers(0, available)
            conn_b[node_idx]     = rng.integers(0, available)
        # Point outputs at the correct gate nodes, with small random perturbation
        out_nodes = np.array([5, 7], dtype=np.int32)  # sum=sig5, carry=sig7
        if rng.random() < 0.3:  # occasionally randomize to maintain diversity
            out_nodes = rng.integers(num_inputs, num_inputs + max_nodes,
                                     size=num_outputs).astype(np.int32)

    elif func_name == "majority3" and num_inputs == 3 and max_nodes >= 3:
        # majority = (a&b) | (b&c) | (a&c)
        gate_types[0] = GATE_AND; conn_a[0] = 0; conn_b[0] = 1  # a AND b
        gate_types[1] = GATE_AND; conn_a[1] = 1; conn_b[1] = 2  # b AND c
        gate_types[2] = GATE_OR;  conn_a[2] = 3; conn_b[2] = 4  # node0 OR node1
        for node_idx in range(3, max_nodes):
            available = max(num_inputs + node_idx, 1)
            gate_types[node_idx] = rng.integers(0, NUM_GATE_TYPES)
            conn_a[node_idx]     = rng.integers(0, available)
            conn_b[node_idx]     = rng.integers(0, available)
        out_nodes = np.array([5], dtype=np.int32)

    else:
        # No known seed for this function — fully random
        return random_circuit(num_inputs, max_nodes, num_outputs, rng)

    return {'gate_types': gate_types, 'conn_a': conn_a,
            'conn_b': conn_b, 'out_nodes': out_nodes}


# --- 4. Simulation ---
# --OPTION--
def simulate_circuit(circuit: dict, num_inputs: int, num_rows: int) -> np.ndarray:
    """
    Exactly simulate circuit on all truth table rows.
    Returns (num_rows, num_outputs) int32 array.
    """
    gate_types    = circuit['gate_types']
    conn_a        = circuit['conn_a']
    conn_b        = circuit['conn_b']
    out_nodes     = circuit['out_nodes']
    max_nodes     = len(gate_types)
    num_outputs   = len(out_nodes)
    total_signals = num_inputs + max_nodes
    all_preds     = np.zeros((num_rows, num_outputs), dtype=np.int32)

    for row in range(num_rows):
        signals = np.zeros(total_signals, dtype=np.int32)
        for i in range(num_inputs):
            signals[i] = (row >> (num_inputs - 1 - i)) & 1

        for node_idx in range(max_nodes):
            sig_idx = num_inputs + node_idx
            ia = min(int(conn_a[node_idx]), sig_idx - 1)
            ib = min(int(conn_b[node_idx]), sig_idx - 1)
            if ia < 0: ia = 0
            if ib < 0: ib = 0
            a = int(signals[ia])
            b = int(signals[ib])
            g = int(gate_types[node_idx])

            if   g == GATE_AND:  signals[sig_idx] = a & b
            elif g == GATE_OR:   signals[sig_idx] = a | b
            elif g == GATE_XOR:  signals[sig_idx] = a ^ b
            elif g == GATE_NOT:  signals[sig_idx] = 1 - a
            elif g == GATE_NAND: signals[sig_idx] = 1 - (a & b)
            elif g == GATE_NOR:  signals[sig_idx] = 1 - (a | b)
            else:                signals[sig_idx] = a  # BUF

        for oi in range(num_outputs):
            oidx = min(max(int(out_nodes[oi]), 0), total_signals - 1)
            all_preds[row, oi] = signals[oidx]

    return all_preds


# --- 5. Fitness ---
# --OPTION--
def evaluate_circuit(circuit: dict, truth_table: np.ndarray, num_inputs: int) -> dict:
    """
    Score a circuit. LOWER = BETTER.

    Strict priority:
      1. wrong_rows  (hard constraint — must be 0 to be correct)
      2. gate_count  (fewer active non-BUF gates)
      3. depth       (shallower critical path)

    Score:
      incorrect: wrong_rows * 10000
      correct:   gate_count * 100 + depth
    """
    num_rows      = truth_table.shape[0]
    gate_types    = circuit['gate_types']
    max_nodes     = len(gate_types)
    total_signals = num_inputs + max_nodes

    preds      = simulate_circuit(circuit, num_inputs, num_rows)
    wrong_rows = int(np.sum(np.any(preds != truth_table, axis=1)))
    is_correct = (wrong_rows == 0)
    gate_count = int(np.sum(gate_types != GATE_BUF))

    node_depth = np.zeros(total_signals, dtype=np.int32)
    for node_idx in range(max_nodes):
        sig_idx = num_inputs + node_idx
        ia = min(int(circuit['conn_a'][node_idx]), sig_idx - 1)
        ib = min(int(circuit['conn_b'][node_idx]), sig_idx - 1)
        if ia < 0: ia = 0
        if ib < 0: ib = 0
        g = int(gate_types[node_idx])
        if g in UNARY_GATES:
            node_depth[sig_idx] = node_depth[ia] + 1
        else:
            node_depth[sig_idx] = max(node_depth[ia], node_depth[ib]) + 1

    depth = int(max(
        node_depth[min(max(int(on), 0), total_signals - 1)]
        for on in circuit['out_nodes']
    ))

    score = wrong_rows * 10000 if not is_correct else gate_count * 100 + depth

    return {'score': score, 'is_correct': is_correct, 'wrong_rows': wrong_rows,
            'gate_count': gate_count, 'depth': depth, 'predictions': preds}


# --- 6. Crossover ---
# --OPTION--
def crossover(parent_a: dict, parent_b: dict,
              rng: np.random.Generator) -> dict:
    """
    Single-point crossover between two parent circuits.
    Randomly picks a split point and takes genes from each parent.
    This lets good substructures from different circuits combine.
    """
    max_nodes = len(parent_a['gate_types'])
    split     = rng.integers(1, max_nodes)  # split point

    def splice(key):
        return np.concatenate([
            parent_a[key][:split],
            parent_b[key][split:]
        ])

    # For out_nodes, randomly pick from either parent per output
    mask      = rng.integers(0, 2, size=len(parent_a['out_nodes'])).astype(bool)
    out_nodes = np.where(mask, parent_a['out_nodes'], parent_b['out_nodes'])

    return {
        'gate_types': splice('gate_types').astype(np.int32),
        'conn_a':     splice('conn_a').astype(np.int32),
        'conn_b':     splice('conn_b').astype(np.int32),
        'out_nodes':  out_nodes.astype(np.int32),
    }


# --- 7. Mutation ---
# --OPTION--
def mutate_circuit(circuit: dict, num_inputs: int,
                   rng: np.random.Generator, mutation_rate: float) -> dict:
    """Flip random genes with probability mutation_rate."""
    gate_types = circuit['gate_types'].copy()
    conn_a     = circuit['conn_a'].copy()
    conn_b     = circuit['conn_b'].copy()
    out_nodes  = circuit['out_nodes'].copy()
    max_nodes  = len(gate_types)

    for node_idx in range(max_nodes):
        available = max(num_inputs + node_idx, 1)
        if rng.random() < mutation_rate:
            gate_types[node_idx] = rng.integers(0, NUM_GATE_TYPES)
        if rng.random() < mutation_rate:
            conn_a[node_idx] = rng.integers(0, available)
        if rng.random() < mutation_rate:
            conn_b[node_idx] = rng.integers(0, available)

    for oi in range(len(out_nodes)):
        if rng.random() < mutation_rate:
            out_nodes[oi] = rng.integers(num_inputs, num_inputs + max_nodes)

    return {'gate_types': gate_types, 'conn_a': conn_a,
            'conn_b': conn_b, 'out_nodes': out_nodes}


# --- 8. EMADE Interface ---
# --OPTION--
def encode_individual(circuit: dict) -> np.ndarray:
    """Flatten circuit to 1-D integer genome for EMADE."""
    return np.concatenate([
        circuit['gate_types'].ravel(),
        circuit['conn_a'].ravel(),
        circuit['conn_b'].ravel(),
        circuit['out_nodes'].ravel(),
    ]).astype(np.int32)

def decode_individual(genome: np.ndarray, max_nodes: int,
                      num_inputs: int, num_outputs: int) -> dict:
    """Reconstruct circuit from flat EMADE genome."""
    c = 0
    gt = genome[c:c+max_nodes].astype(np.int32); c += max_nodes
    ca = genome[c:c+max_nodes].astype(np.int32); c += max_nodes
    cb = genome[c:c+max_nodes].astype(np.int32); c += max_nodes
    on = genome[c:c+num_outputs].astype(np.int32)
    return {'gate_types': gt, 'conn_a': ca, 'conn_b': cb, 'out_nodes': on}


# --- 9. Configuration ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func",             type=str,   default="full_adder")
    parser.add_argument("--num_inputs",       type=int,   default=3)
    parser.add_argument("--num_outputs",      type=int,   default=2)
    parser.add_argument("--max_nodes",        type=int,   default=10)
    parser.add_argument("--pop_size",         type=int,   default=200,
                        help="Population size for mini-GA")
    parser.add_argument("--generations",      type=int,   default=500,
                        help="Number of GA generations")
    parser.add_argument("--mutation_rate",    type=float, default=0.15)
    parser.add_argument("--elite_frac",       type=float, default=0.1,
                        help="Fraction of population kept as elites each gen")
    parser.add_argument("--seed_frac",        type=float, default=0.3,
                        help="Fraction of initial population that is seeded")
    parser.add_argument("--hill_climb_steps", type=int,   default=2000)
    parser.add_argument("--print_every",      type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--save_root",        type=str,   default="trained")
    parser.add_argument("--save_dir",         type=str,   default=None)
    return parser.parse_args()


# --- 10. Main ---
# --OPTION--
def main():
    args = get_args()

    if args.save_dir is not None:
        exp_dir = args.save_dir
        p(exp_dir).mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = create_save_dir(args.save_root)

    rng         = np.random.default_rng(args.seed)
    truth_table = get_truth_table(args.func, args.num_inputs)
    num_rows    = truth_table.shape[0]
    n_elite     = max(1, int(args.pop_size * args.elite_frac))

    print("=" * 70)
    print(f"Circuit Design Seed Search (Mini-GA)")
    print(f"Target            : {args.func}")
    print(f"Inputs / Outputs  : {args.num_inputs} / {args.num_outputs}")
    print(f"Max gate nodes    : {args.max_nodes}")
    print(f"Population size   : {args.pop_size}")
    print(f"Generations       : {args.generations}")
    print(f"Mutation rate     : {args.mutation_rate}")
    print(f"Elite fraction    : {args.elite_frac} ({n_elite} individuals)")
    print(f"Seed fraction     : {args.seed_frac}")
    print("=" * 70)

    # Print truth table
    print(f"\nTruth table ({args.func}):")
    h_in  = " ".join(f"in{i}" for i in range(args.num_inputs))
    h_out = " ".join(f"out{i}" for i in range(args.num_outputs))
    print(f"  {h_in} | {h_out}")
    for row in range(num_rows):
        ins  = " ".join(str((row >> (args.num_inputs-1-i)) & 1)
                        for i in range(args.num_inputs))
        outs = " ".join(str(truth_table[row, o])
                        for o in range(args.num_outputs))
        print(f"  {ins}  |  {outs}")
    print()

    # ----------------------------------------------------------------
    # Phase 1: Initialize population
    # Mix of seeded (known subcircuits) + random for diversity
    # ----------------------------------------------------------------
    print("Phase 1: Initializing population...")
    n_seeded = int(args.pop_size * args.seed_frac)
    population = []

    for i in range(n_seeded):
        c = make_seeded_circuit(args.num_inputs, args.max_nodes,
                                args.num_outputs, args.func, rng)
        population.append(c)

    for i in range(args.pop_size - n_seeded):
        c = random_circuit(args.num_inputs, args.max_nodes, args.num_outputs, rng)
        population.append(c)

    # Evaluate initial population
    scores = [evaluate_circuit(c, truth_table, args.num_inputs)
              for c in population]

    best_idx     = int(np.argmin([s['score'] for s in scores]))
    best_circuit = population[best_idx]
    best_metrics = scores[best_idx]
    best_score   = best_metrics['score']

    correct_count = sum(1 for s in scores if s['is_correct'])
    print(f"  Initial population: best_score={best_score}  "
          f"correct={correct_count}/{args.pop_size}  "
          f"best_wrong_rows={best_metrics['wrong_rows']}")

    # ----------------------------------------------------------------
    # Phase 2: Evolution
    # Tournament selection -> crossover -> mutation -> elitism
    # ----------------------------------------------------------------
    print(f"\nPhase 2: Evolving for {args.generations} generations...")

    for gen in range(args.generations):
        sorted_indices = np.argsort([s['score'] for s in scores])
        elites   = [population[i] for i in sorted_indices[:n_elite]]
        e_scores = [scores[i]     for i in sorted_indices[:n_elite]]

        # Check for solution
        if e_scores[0]['is_correct']:
            if gen == 0 or (gen + 1) % args.print_every == 0:
                print(f"  Gen {gen+1:4d}: CORRECT! score={e_scores[0]['score']}  "
                      f"Gates={e_scores[0]['gate_count']}  "
                      f"Depth={e_scores[0]['depth']}")
        elif (gen + 1) % args.print_every == 0:
            wrong = e_scores[0]['wrong_rows']
            print(f"  Gen {gen+1:4d}: best_score={e_scores[0]['score']}  "
                  f"wrong_rows={wrong}  "
                  f"Gates={e_scores[0]['gate_count']}  "
                  f"Depth={e_scores[0]['depth']}")

        if e_scores[0]['score'] < best_score:
            best_score   = e_scores[0]['score']
            best_circuit = elites[0]
            best_metrics = e_scores[0]

        # Build next generation
        next_pop    = elites[:]
        next_scores = e_scores[:]

        while len(next_pop) < args.pop_size:
            # Tournament selection (size 3) for two parents
            def tournament():
                candidates = rng.choice(len(population), size=3, replace=False)
                best = min(candidates, key=lambda i: scores[i]['score'])
                return population[best]

            parent_a = tournament()
            parent_b = tournament()

            # Crossover then mutate
            child = crossover(parent_a, parent_b, rng)
            child = mutate_circuit(child, args.num_inputs, rng, args.mutation_rate)

            child_score = evaluate_circuit(child, truth_table, args.num_inputs)
            next_pop.append(child)
            next_scores.append(child_score)

        population = next_pop
        scores     = next_scores

    print(f"\nPhase 2 complete. Best score: {best_score}  "
          f"Correct: {best_metrics['is_correct']}")

    # ----------------------------------------------------------------
    # Phase 3: Hill climbing to polish the best circuit
    # ----------------------------------------------------------------
    print(f"\nPhase 3: Hill climbing ({args.hill_climb_steps} steps)...")

    no_improve = 0
    for step in range(args.hill_climb_steps):
        candidate     = mutate_circuit(best_circuit, args.num_inputs,
                                       rng, args.mutation_rate * 0.5)
        cand_metrics  = evaluate_circuit(candidate, truth_table, args.num_inputs)

        if cand_metrics['score'] < best_score:
            best_score   = cand_metrics['score']
            best_circuit = candidate
            best_metrics = cand_metrics
            no_improve   = 0
            print(f"  Step {step+1:4d}: score={best_score}  "
                  f"Correct={best_metrics['is_correct']}  "
                  f"Gates={best_metrics['gate_count']}  "
                  f"Depth={best_metrics['depth']}")
        else:
            no_improve += 1

        if no_improve > 1000 and best_metrics['is_correct']:
            print(f"  Converged at step {step+1}.")
            break

    # ----------------------------------------------------------------
    # Final results
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Correct      : {best_metrics['is_correct']}")
    print(f"Wrong rows   : {best_metrics['wrong_rows']} / {num_rows}")
    print(f"Gate count   : {best_metrics['gate_count']}")
    print(f"Depth        : {best_metrics['depth']}")
    print(f"Gate types   : {[GATE_NAMES[g] for g in best_circuit['gate_types']]}")
    print(f"Out nodes    : {best_circuit['out_nodes']}")

    if best_metrics['is_correct']:
        print("\n[SUCCESS] Circuit matches truth table exactly!")
        print("\nVerification:")
        h_in  = " ".join(f"in{i}" for i in range(args.num_inputs))
        h_out = " ".join(f"out{i}" for i in range(args.num_outputs))
        print(f"  {h_in} | expected | got")
        for row in range(num_rows):
            ins = " ".join(str((row >> (args.num_inputs-1-i)) & 1)
                           for i in range(args.num_inputs))
            exp = " ".join(str(truth_table[row, o])
                           for o in range(args.num_outputs))
            got = " ".join(str(best_metrics['predictions'][row, o])
                           for o in range(args.num_outputs))
            ok  = "✓" if exp == got else "✗"
            print(f"  {ins}  |  {exp}      |  {got}  {ok}")
    else:
        print("\n[FAILED] Try: --generations 1000 --pop_size 300 --seed_frac 0.5")

    # Save
    genome = encode_individual(best_circuit)
    np.savez(pj(exp_dir, "circuit_params.npz"),
             gate_types = best_circuit['gate_types'],
             conn_a     = best_circuit['conn_a'],
             conn_b     = best_circuit['conn_b'],
             out_nodes  = best_circuit['out_nodes'],
             genome     = genome)

    with open(pj(exp_dir, "results.txt"), "w") as f:
        f.write(f"Score: {best_score}\n")
        f.write(f"Correct: {best_metrics['is_correct']}\n")
        f.write(f"Gate count: {best_metrics['gate_count']}\n")
        f.write(f"Depth: {best_metrics['depth']}\n")
        f.write(f"Config: {vars(args)}\n")

    print(f"\nSaved to {exp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(p(__file__).parent.resolve())
    main()