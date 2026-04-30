"""
check_benchmarks.py

Scans the `results/` folder for *_rank.txt files produced by evalDynamic.py
and reports which models achieve a valid decomposition (best_valid_R is not None)
at or below the best-known rank and/or the AlphaEvolve rank for each matrix size.

Usage:
    python check_benchmarks.py [--results_dir results]
"""

import argparse
import re
import sys
from pathlib import Path

# ============================================================
# BENCHMARK TABLE
# ⟨m, n, p⟩ -> (best_known_rank, alphaevolve_rank)
# From the AlphaEvolve paper screenshot.
# All entries where AlphaEvolve improved (bold) or matched are included.
# ============================================================
BENCHMARKS = {
    (2, 4, 5): (33, 32),
    (2, 4, 7): (46, 45),
    (2, 4, 8): (52, 51),
    (2, 5, 6): (48, 47),
    (3, 3, 3): (23, 23),
    (3, 4, 6): (56, 54),
    (3, 4, 7): (66, 63),
    (3, 4, 8): (75, 74),
    (3, 5, 6): (70, 68),
    (3, 5, 7): (82, 80),
    (4, 4, 4): (49, 48),
    (4, 4, 5): (62, 61),
    (4, 4, 7): (87, 85),
    (4, 4, 8): (98, 96),
    (4, 5, 6): (93, 90),
    (5, 5, 5): (93, 93),
    # ---- sizes not in the AlphaEvolve table but your eval might search ----
    # For these, best_known == alphaevolve (no improvement claimed).
    # Add more here as needed:
    (2, 2, 2): (7,  7),   # Strassen
    (2, 3, 3): (11, 11),
    (3, 3, 4): (29, 29),
}

# ============================================================
# Regex patterns for the rank log lines produced by evalDynamic.py
# Example line:
#   size: 2x2x2  best_valid_R: 7  rank_ratio: 0.8750  valid: True
# ============================================================
LINE_RE = re.compile(
    r"size:\s*(\d+)x(\d+)x(\d+)"
    r"\s+best_valid_R:\s*(\S+)"
    r"\s+rank_ratio:\s*([\d.]+)"
    r"\s+valid:\s*(True|False)"
)
GENE_RE = re.compile(r"gene_id:\s*(\S+)")


def parse_rank_file(path: Path):
    """
    Parse a *_rank.txt file.
    Returns (gene_id, list_of_size_dicts).
    Each size dict: {N, M, P, best_valid_R (int or None), rank_ratio, valid}
    """
    gene_id = path.stem.replace("_rank", "")
    sizes = []
    with open(path) as f:
        for line in f:
            m = GENE_RE.search(line)
            if m:
                gene_id = m.group(1)
                continue
            m = LINE_RE.search(line)
            if m:
                N, M, P = int(m.group(1)), int(m.group(2)), int(m.group(3))
                raw_r = m.group(4)
                best_valid_R = None if raw_r in ("None", "null", "") else int(raw_r)
                rank_ratio = float(m.group(5))
                valid = m.group(6) == "True"
                sizes.append({
                    "N": N, "M": M, "P": P,
                    "best_valid_R": best_valid_R,
                    "rank_ratio": rank_ratio,
                    "valid": valid,
                })
    return gene_id, sizes


def classify(N, M, P, best_valid_R):
    """
    Given dimensions N,M,P and a validated rank (or None),
    return (eq_bk, lt_bk, eq_ae, lt_ae, lt_triv, best_known_rank, ae_rank, trivial_rank).
    eq_bk: best_valid_R == best_known_rank
    lt_bk: best_valid_R < best_known_rank
    eq_ae: best_valid_R == alphaevolve_rank
    lt_ae: best_valid_R < alphaevolve_rank
    lt_triv: best_valid_R < trivial (n*m*p)
    """
    size_key = (N, M, P)
    bk = ae = None
    if size_key in BENCHMARKS:
        bk, ae = BENCHMARKS[size_key]
    trivial = N * M * P
    if best_valid_R is None:
        return False, False, False, False, False, bk, ae, trivial
    eq_bk = bk is not None and best_valid_R == bk
    lt_bk = bk is not None and best_valid_R < bk
    eq_ae = ae is not None and best_valid_R == ae
    lt_ae = ae is not None and best_valid_R < ae
    lt_triv = best_valid_R < trivial
    return eq_bk, lt_bk, eq_ae, lt_ae, lt_triv, bk, ae, trivial


def main():
    parser = argparse.ArgumentParser(
        description="Scan results/ for models matching best-known / AlphaEvolve ranks."
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Directory containing *_rank.txt files (default: results/)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    rank_files = sorted(results_dir.glob("*_rank.txt"))
    if not rank_files:
        print(f"[WARNING] No *_rank.txt files found in {results_dir}")
        sys.exit(0)

    # ---- Collect all hits ----
    rows = []
    matches_bk = []   # (gene_id, size, R, bk_rank)
    beats_bk   = []   # (gene_id, size, R, bk_rank, ae_rank)
    matches_ae = []   # (gene_id, size, R, ae_rank)
    beats_ae   = []   # (gene_id, size, R, ae_rank)
    beats_triv = []   # (gene_id, size, R, trivial_rank)

    for rf in rank_files:
        gene_id, sizes = parse_rank_file(rf)
        for s in sizes:
            size_str = f"{s['N']}x{s['M']}x{s['P']}"
            eq_bk, lt_bk, eq_ae, lt_ae, lt_triv, bk, ae, trivial = \
                classify(s["N"], s["M"], s["P"], s["best_valid_R"]) 
            if lt_triv:
                beats_triv.append((gene_id, size_str, s["best_valid_R"], trivial))
            if eq_bk:
                matches_bk.append((gene_id, size_str, s["best_valid_R"], bk))
            if lt_bk:
                beats_bk.append((gene_id, size_str, s["best_valid_R"], bk, ae))
            if eq_ae:
                matches_ae.append((gene_id, size_str, s["best_valid_R"], ae))
            if lt_ae:
                beats_ae.append((gene_id, size_str, s["best_valid_R"], ae))

    # ---- Print summary ----
    W = 80
    print("=" * W)
    print("BENCHMARK HIT REPORT")
    print("=" * W)

    print(f"\nMODELS BEATING TRIVIAL RANK (n*m*p)  ({len(beats_triv)} hits)")
    print("-" * W)
    if beats_triv:
        for gene_id, size, R, trivial in beats_triv:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (trivial={trivial})")
    else:
        print("  (none)")

    print(f"\nMODELS MATCHING BEST-KNOWN RANK  ({len(matches_bk)} hits)")
    print("-" * W)
    if matches_bk:
        for gene_id, size, R, bk in matches_bk:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (best_known={bk})")
    else:
        print("  (none)")

    print(f"\nMODELS BEATING BEST-KNOWN RANK  ({len(beats_bk)} hits)")
    print("-" * W)
    if beats_bk:
        for gene_id, size, R, bk, ae in beats_bk:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (best_known={bk}, alphaevolve={ae})")
    else:
        print("  (none)")

    print(f"\nMODELS MATCHING ALPHAEVOLVE RANK  ({len(matches_ae)} hits)")
    print("-" * W)
    if matches_ae:
        for gene_id, size, R, ae in matches_ae:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (alphaevolve={ae})")
    else:
        print("  (none)")

    print(f"\nMODELS BEATING ALPHAEVOLVE RANK  ({len(beats_ae)} hits)")
    print("-" * W)
    if beats_ae:
        for gene_id, size, R, ae in beats_ae:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (alphaevolve={ae})")
    else:
        print("  (none)")

    print("=" * W)


if __name__ == "__main__":
    main()