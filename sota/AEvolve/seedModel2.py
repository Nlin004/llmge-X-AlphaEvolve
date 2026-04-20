import argparse
import os
import json
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path as p
from os.path import join as pj
from typing import List, Tuple, Dict, Optional

# -----------------------------
# 1) Utilities for Directory Management
# -----------------------------

def create_save_dir(save_root: str) -> str:
    save_root_path = p(save_root)
    save_root_path.mkdir(exist_ok=True, parents=True)
    nums = []
    for exp_dir in save_root_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("exp") and exp_dir.name[3:].isdigit():
            nums.append(int(exp_dir.name[3:]))
    save_dir = pj(save_root, f"exp{(max(nums) + 1) if nums else 1}")
    p(save_dir).mkdir(exist_ok=True, parents=True)
    return save_dir


# -----------------------------
# 2) Bitset truth-table evaluation helpers
# -----------------------------
# Boolean functions over n inputs are stored as a bitmask of length 2^n.
# Bit i = output for input assignment i, where bit v of i = value of input x_v.

def var_mask(n: int, var_idx: int) -> int:
    """Bitmask for variable x[var_idx] over all 2^n assignments."""
    m = 0
    for a in range(1 << n):
        if (a >> var_idx) & 1:
            m |= (1 << a)
    return m

def full_mask(n: int) -> int:
    return (1 << (1 << n)) - 1

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# -----------------------------
# 3) Circuit representation
# -----------------------------

OPS = ("NOT", "NAND", "AND", "OR", "XOR", "MUX")

@dataclass
class Node:
    op: str
    a: int
    b: int
    c: int = -1   # MUX only

@dataclass
class Circuit:
    n_inputs: int
    nodes: List[Node]
    output_idx: int
    allow_ops: Tuple[str, ...] = OPS
    gate_count: int = 0

    def __post_init__(self):
        self.gate_count = len(self.nodes)

    def clone(self) -> "Circuit":
        return Circuit(
            n_inputs=self.n_inputs,
            nodes=[Node(nd.op, nd.a, nd.b, nd.c) for nd in self.nodes],
            output_idx=self.output_idx,
            allow_ops=self.allow_ops,
        )


def eval_circuit_bitset(circ: Circuit) -> int:
    """
    Evaluate circuit over all input assignments as a bitmask int.

    FIX: All fanin signal reads and output_idx are clamped to the
    currently available signal range, preventing IndexError when
    mutate() produces temporarily out-of-range indices.
    """
    n = circ.n_inputs
    M = full_mask(n)
    signals: List[int] = [var_mask(n, i) for i in range(n)]

    for nd in circ.nodes:
        def gs(idx: int) -> int:
            return signals[min(max(idx, 0), len(signals) - 1)]

        if nd.op == "NOT":
            signals.append((~gs(nd.a)) & M)
        elif nd.op == "NAND":
            signals.append((~(gs(nd.a) & gs(nd.b))) & M)
        elif nd.op == "AND":
            signals.append(gs(nd.a) & gs(nd.b))
        elif nd.op == "OR":
            signals.append(gs(nd.a) | gs(nd.b))
        elif nd.op == "XOR":
            signals.append(gs(nd.a) ^ gs(nd.b))
        elif nd.op == "MUX":
            sel, t, f = gs(nd.a), gs(nd.b), gs(nd.c)
            signals.append((sel & t) | ((~sel) & f & M))
        else:
            raise ValueError(f"Unknown op {nd.op}")

    oidx = min(max(circ.output_idx, 0), len(signals) - 1)
    return signals[oidx]


def approx_depth(circ: Circuit) -> int:
    """
    Depth estimate via DP. Inputs = depth 0, each gate = 1 + max fanin depth.
    FIX: depth lookups are clamped to valid range.
    """
    depths = [0] * circ.n_inputs
    for nd in circ.nodes:
        def gd(idx: int) -> int:
            return depths[min(max(idx, 0), len(depths) - 1)]

        if nd.op == "NOT":
            depths.append(1 + gd(nd.a))
        elif nd.op == "MUX":
            depths.append(1 + max(gd(nd.a), gd(nd.b), gd(nd.c)))
        else:
            depths.append(1 + max(gd(nd.a), gd(nd.b)))

    oidx = min(max(circ.output_idx, 0), len(depths) - 1)
    return depths[oidx]


# -----------------------------
# 4) Seed generators
# -----------------------------

def sop_seed(n: int, target: int) -> Circuit:
    """
    Sum-of-Products seed using AND/OR/NOT.
    Always produces a provably correct circuit for any target truth table.
    The GA then minimizes gate count and depth from this starting point.
    """
    nodes: List[Node] = []
    sc = n

    def add(op: str, a: int, b: int, c: int = -1) -> int:
        nonlocal sc
        nodes.append(Node(op, a, b, c))
        sc += 1
        return sc - 1

    not_cache: Dict[int, int] = {}

    def get_not(x: int) -> int:
        if x not in not_cache:
            not_cache[x] = add("NOT", x, -1)
        return not_cache[x]

    minterms = [a for a in range(1 << n) if (target >> a) & 1]

    if len(minterms) == 0:
        out = add("AND", 0, get_not(0))
        return Circuit(n_inputs=n, nodes=nodes, output_idx=out, allow_ops=OPS)

    if len(minterms) == (1 << n):
        out = add("OR", 0, get_not(0))
        return Circuit(n_inputs=n, nodes=nodes, output_idx=out, allow_ops=OPS)

    term_signals = []
    for a in minterms:
        lits = [v if (a >> v) & 1 else get_not(v) for v in range(n)]
        term = lits[0]
        for li in lits[1:]:
            term = add("AND", term, li)
        term_signals.append(term)

    out = term_signals[0]
    for ts in term_signals[1:]:
        out = add("OR", out, ts)

    return Circuit(n_inputs=n, nodes=nodes, output_idx=out, allow_ops=OPS)


def nand_only_from_sop(n: int, target: int) -> Circuit:
    """SOP realized with NAND/NOT only. Different gate basis = more diversity."""
    nodes: List[Node] = []
    sc = n

    def add_not(sig: int) -> int:
        nonlocal sc; nodes.append(Node("NOT", sig, -1)); sc += 1; return sc - 1

    def add_nand(a: int, b: int) -> int:
        nonlocal sc; nodes.append(Node("NAND", a, b)); sc += 1; return sc - 1

    not_cache: Dict[int, int] = {}

    def lit(var: int, pos: bool) -> int:
        if pos: return var
        if var not in not_cache: not_cache[var] = add_not(var)
        return not_cache[var]

    def add_and(a: int, b: int) -> int:
        t = add_nand(a, b); return add_nand(t, t)

    def add_or(a: int, b: int) -> int:
        return add_not(add_and(add_not(a), add_not(b)))

    minterms = [a for a in range(1 << n) if (target >> a) & 1]

    if len(minterms) == 0:
        nx = lit(0, False); t = add_nand(0, nx); c0 = add_nand(t, t)
        return Circuit(n_inputs=n, nodes=nodes, output_idx=c0, allow_ops=OPS)

    if len(minterms) == (1 << n):
        nx = lit(0, False); c1 = add_nand(0, nx)
        return Circuit(n_inputs=n, nodes=nodes, output_idx=c1, allow_ops=OPS)

    term_signals = []
    for a in minterms:
        lits = [lit(v, bool((a >> v) & 1)) for v in range(n)]
        term = lits[0]
        for li in lits[1:]: term = add_and(term, li)
        term_signals.append(term)

    out = term_signals[0]
    for ts in term_signals[1:]: out = add_or(out, ts)
    return Circuit(n_inputs=n, nodes=nodes, output_idx=out, allow_ops=OPS)


def mux_tree_seed(n: int, target: int, var_order: Optional[List[int]] = None) -> Circuit:
    """Shannon expansion into a MUX tree. f = MUX(x, f|x=1, f|x=0)."""
    if var_order is None:
        var_order = list(range(n))

    nodes: List[Node] = []
    sc = n
    full_table = [(target >> a) & 1 for a in range(1 << n)]

    def add_mux(sel: int, t_sig: int, f_sig: int) -> int:
        nonlocal sc; nodes.append(Node("MUX", sel, t_sig, f_sig)); sc += 1; return sc - 1

    def add_const(bit: int) -> int:
        nonlocal sc
        nodes.append(Node("NOT", 0, -1)); sc += 1; nx0 = sc - 1
        if bit == 1: nodes.append(Node("OR",  0, nx0))
        else:        nodes.append(Node("AND", 0, nx0))
        sc += 1; return sc - 1

    def build(indices: List[int], remaining: List[int]) -> int:
        vals = [full_table[i] for i in indices]
        if all(v == 0 for v in vals): return add_const(0)
        if all(v == 1 for v in vals): return add_const(1)
        if not remaining:             return add_const(vals[0])
        v    = remaining[0]
        idx0 = [i for i in indices if ((i >> v) & 1) == 0]
        idx1 = [i for i in indices if ((i >> v) & 1) == 1]
        f0   = build(idx0, remaining[1:])
        f1   = build(idx1, remaining[1:])
        return add_mux(v, f1, f0)

    out = build(list(range(1 << n)), var_order)
    return Circuit(n_inputs=n, nodes=nodes, output_idx=out, allow_ops=OPS)


def seed_random_aig(n: int, max_gates: int, rng: random.Random) -> Circuit:
    """Random AIG (AND/NOT only) seed for population diversity."""
    nodes: List[Node] = []
    for _ in range(max_gates):
        op  = rng.choice(["AND", "NOT"])
        cur = max(n + len(nodes), 1)
        if op == "NOT": nodes.append(Node("NOT", rng.randrange(cur), -1))
        else:           nodes.append(Node("AND", rng.randrange(cur), rng.randrange(cur)))
    return Circuit(n_inputs=n, nodes=nodes, output_idx=n + len(nodes) - 1, allow_ops=OPS)


# -----------------------------
# 5) Mutation operators
# -----------------------------

def mutate(circ: Circuit, rng: random.Random, max_gates: int) -> Circuit:
    """
    Small random edits: tweak op, rewire fanins, insert node, delete node,
    or change output tap.

    FIX 1 (delete branch): output_idx is retapped when it points AT OR
    AFTER the deleted signal — not just exactly equal — because all signals
    at higher indices shift down by 1 after the pop.

    FIX 2 (delete branch): output_idx is clamped to the new valid range
    after all fixup steps complete.
    """
    child = circ.clone()
    n     = child.n_inputs

    def rand_sig() -> int:
        return rng.randrange(max(1, n + len(child.nodes)))

    if len(child.nodes) == 0:
        child.nodes.append(Node("NOT", 0, -1))
        child.output_idx = n
        return child

    move = rng.random()

    # 1) Tweak a random node's op or fanins
    if move < 0.45:
        i  = rng.randrange(len(child.nodes))
        op = rng.choice(child.allow_ops)
        if op == "NOT":
            child.nodes[i] = Node("NOT", rand_sig(), -1)
        elif op == "MUX":
            child.nodes[i] = Node("MUX", rand_sig(), rand_sig(), rand_sig())
        else:
            child.nodes[i] = Node(op, rand_sig(), rand_sig())

    # 2) Insert a new node at the end
    elif move < 0.70 and len(child.nodes) < max_gates:
        op = rng.choice(child.allow_ops)
        if op == "NOT":
            child.nodes.append(Node("NOT", rand_sig(), -1))
        elif op == "MUX":
            child.nodes.append(Node("MUX", rand_sig(), rand_sig(), rand_sig()))
        else:
            child.nodes.append(Node(op, rand_sig(), rand_sig()))
        if rng.random() < 0.35:
            child.output_idx = n + len(child.nodes) - 1

    # 3) Delete a random node
    elif move < 0.85 and len(child.nodes) > 1:
        del_idx = rng.randrange(len(child.nodes))
        del_sig = n + del_idx

        # FIX 1: retap if output points at or after the deleted signal
        if child.output_idx >= del_sig:
            child.output_idx = rng.randrange(max(1, n + len(child.nodes) - 1))

        child.nodes.pop(del_idx)

        def fix_sig(s: int) -> int:
            if s < n:
                return s                       # primary input, unaffected
            node_idx = s - n
            if node_idx == del_idx:
                return rng.randrange(n)        # deleted node — fall back to input
            if node_idx > del_idx:
                return s - 1                   # shift down
            return s

        for j, nd in enumerate(child.nodes):
            nd.a = fix_sig(nd.a)
            if nd.op != "NOT": nd.b = fix_sig(nd.b)
            if nd.op == "MUX": nd.c = fix_sig(nd.c)
            child.nodes[j] = nd

        # FIX 2: clamp output_idx after all shifts
        child.output_idx = min(child.output_idx, n + len(child.nodes) - 1)

    # 4) Change output tap
    else:
        child.output_idx = rand_sig()

    child.gate_count = len(child.nodes)
    return child


# -----------------------------
# 6) Fitness + verification
# -----------------------------

@dataclass
class Fitness:
    error: int
    gates: int
    depth: int
    score: float


def score_circuit(circ: Circuit, target: int,
                  w_gates: float, w_depth: float) -> Fitness:
    y     = eval_circuit_bitset(circ)
    err   = hamming(y, target)
    g     = circ.gate_count
    d     = approx_depth(circ)
    score = float(err + w_gates * g + w_depth * d)
    return Fitness(error=err, gates=g, depth=d, score=score)


def verify_exact(circ: Circuit, target: int) -> bool:
    return eval_circuit_bitset(circ) == target


# -----------------------------
# 7) CLI + Main
# -----------------------------

def parse_target_from_args(n: int,
                            target_hex: Optional[str],
                            target_file: Optional[str]) -> int:
    """
    Parse the target truth table from CLI args.

    The target is a bitmask of 2^n bits where bit i = f(assignment i).
    Assignment i: bit v of i = value of input x_v (LSB = x0).

    Common values:
      Full adder SUM   (n=3): 0x96
      Full adder CARRY (n=3): 0xe8
      Majority3        (n=3): 0xe8
      XOR4 / Parity4   (n=4): 0x6996
    """
    if target_file is not None:
        with open(target_file) as f:
            data = json.load(f)
        if "target_hex" in data:
            return int(data["target_hex"], 16)
        if "target_bits" in data:
            bits = data["target_bits"]
            if len(bits) != (1 << n):
                raise ValueError(f"target_bits must have length {1 << n}")
            t = 0
            for i, b in enumerate(bits):
                t |= (int(b) & 1) << i
            return t
        raise ValueError("JSON must contain target_hex or target_bits")

    if target_hex is None:
        raise ValueError("Provide --target_hex or --target_file")

    return int(target_hex, 16)


def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n_inputs",    type=int,   default=3)
    ap.add_argument("--target_hex",  type=str,   default="0x96",
                    help="Truth table as hex bitmask. "
                         "SUM=0x96, CARRY=0xe8, XOR4=0x6996")
    ap.add_argument("--target_file", type=str,   default=None,
                    help="JSON with target_hex or target_bits (overrides --target_hex)")

    ap.add_argument("--iterations",  type=int,   default=2000)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--elite_frac",  type=float, default=0.10)
    ap.add_argument("--max_gates",   type=int,   default=80)
    ap.add_argument("--seed",        type=int,   default=42)

    ap.add_argument("--w_gates",     type=float, default=0.02)
    ap.add_argument("--w_depth",     type=float, default=0.05)

    ap.add_argument("--save_root",   type=str,   default="trained_circuits")
    ap.add_argument("--save_dir",    type=str,   default=None)
    ap.add_argument("--print_every", type=int,   default=100)

    return ap.parse_args()


def main():
    args = get_args()
    rng  = random.Random(args.seed)

    if args.save_dir is not None:
        exp_dir = args.save_dir
        p(exp_dir).mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = create_save_dir(args.save_root)

    n      = args.n_inputs
    target = parse_target_from_args(n, args.target_hex, args.target_file)

    print("=" * 70)
    print("Circuit Design Seed Model")
    print(f"n_inputs    : {n}")
    print(f"target      : {hex(target)}")
    print(f"population  : {args.batch_size}")
    print(f"iterations  : {args.iterations}")
    print(f"elite_frac  : {args.elite_frac}")
    print(f"max_gates   : {args.max_gates}")
    print(f"weights     : w_gates={args.w_gates}, w_depth={args.w_depth}")
    print(f"exp_dir     : {exp_dir}")
    print("=" * 70)

    print(f"\nTruth table ({hex(target)}, {n} inputs):")
    print("  " + " ".join(f"x{i}" for i in range(n)) + " | f")
    for a in range(1 << n):
        bits = " ".join(str((a >> i) & 1) for i in range(n))
        print(f"  {bits} | {(target >> a) & 1}")
    print()

    # ---- Initialize population ----
    pop: List[Circuit] = []

    # Correct anchors first — sop_seed always produces an exact circuit
    pop.append(sop_seed(n, target))
    pop.append(nand_only_from_sop(n, target))

    order = list(range(n))
    rng.shuffle(order)
    pop.append(mux_tree_seed(n, target, var_order=order))

    # Random AIG seeds for diversity
    while len(pop) < max(8, args.batch_size // 6):
        pop.append(seed_random_aig(n, max_gates=min(20, args.max_gates), rng=rng))

    # Fill rest with mutated variants
    while len(pop) < args.batch_size:
        pop.append(mutate(rng.choice(pop[:]), rng, max_gates=args.max_gates))

    # ---- Evolution loop ----
    elite_k      = max(1, int(args.elite_frac * args.batch_size))
    best_overall: Optional[Circuit] = None
    best_fit:     Optional[Fitness] = None

    for it in range(args.iterations):
        fits = [score_circuit(c, target, args.w_gates, args.w_depth) for c in pop]
        idx  = sorted(range(len(pop)), key=lambda i: fits[i].score)

        elites     = [pop[i]  for i in idx[:elite_k]]
        elite_fits = [fits[i] for i in idx[:elite_k]]

        if best_fit is None or elite_fits[0].score < best_fit.score:
            best_fit     = elite_fits[0]
            best_overall = elites[0].clone()

        if it % args.print_every == 0:
            print(f"Iter {it:5d} | score={best_fit.score:.3f} "
                  f"| err={best_fit.error} | gates={best_fit.gates} "
                  f"| depth={best_fit.depth}")

        if best_fit.error == 0:
            print(f"\n[SOLVED] Exact match at iteration {it}!")
            break

        new_pop = elites[:]
        while len(new_pop) < args.batch_size:
            new_pop.append(mutate(rng.choice(elites), rng, max_gates=args.max_gates))
        pop = new_pop

    # ---- Results ----
    assert best_overall is not None and best_fit is not None
    is_exact = verify_exact(best_overall, target)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Exact match : {is_exact}")
    print(f"Error       : {best_fit.error}")
    print(f"Gates       : {best_fit.gates}")
    print(f"Depth       : {best_fit.depth}")
    print(f"Score       : {best_fit.score:.3f}")

    if is_exact:
        print("\nVerification:")
        print("  " + " ".join(f"x{i}" for i in range(n)) + " | expected | got")
        result_mask = eval_circuit_bitset(best_overall)
        for a in range(1 << n):
            bits = " ".join(str((a >> i) & 1) for i in range(n))
            exp  = (target      >> a) & 1
            got  = (result_mask >> a) & 1
            print(f"  {bits} |    {exp}     |  {got}   {'✓' if exp==got else '✗'}")

    # Save
    circ_file = pj(exp_dir, "best_circuit.json")
    with open(circ_file, "w") as f:
        json.dump({
            "n_inputs":   best_overall.n_inputs,
            "output_idx": best_overall.output_idx,
            "nodes":      [{"op": nd.op, "a": nd.a, "b": nd.b, "c": nd.c}
                           for nd in best_overall.nodes],
            "fitness":    {"score": best_fit.score, "error": best_fit.error,
                           "gates": best_fit.gates, "depth": best_fit.depth},
            "target_hex": hex(target),
        }, f, indent=2)

    results_file = pj(exp_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Score: {best_fit.score}\n")
        f.write(f"Error: {best_fit.error}\n")
        f.write(f"Gates: {best_fit.gates}\n")
        f.write(f"Depth: {best_fit.depth}\n")
        f.write(f"Exact: {is_exact}\n")
        f.write(f"Config: {vars(args)}\n")

    print(f"\nSaved: {circ_file}")
    print(f"Saved: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(p(__file__).parent.resolve())
    main()