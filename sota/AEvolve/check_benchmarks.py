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


def classify(best_valid_R, size_key):
    """
    Given a validated rank (or None) and a size key (N,M,P),
    return (hits_best_known, hits_alphaevolve, best_known_rank, ae_rank).
    """
    if size_key not in BENCHMARKS:
        return False, False, None, None
    bk, ae = BENCHMARKS[size_key]
    if best_valid_R is None:
        return False, False, bk, ae
    hits_bk = best_valid_R <= bk
    hits_ae = best_valid_R <= ae
    return hits_bk, hits_ae, bk, ae


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
    hits_bk   = []     # (gene_id, size, R, bk_rank)
    hits_ae   = []     # (gene_id, size, R, ae_rank)

    for rf in rank_files:
        gene_id, sizes = parse_rank_file(rf)
        for s in sizes:
            key = (s["N"], s["M"], s["P"])
            h_bk, h_ae, bk, ae = classify(s["best_valid_R"], key)
            size_str = f"{s['N']}x{s['M']}x{s['P']}"
            if h_bk:
                hits_bk.append((gene_id, size_str, s["best_valid_R"], bk, ae))
            if h_ae:
                hits_ae.append((gene_id, size_str, s["best_valid_R"], ae))

    # ---- Print summary ----
    W = 80
    print("=" * W)
    print("BENCHMARK HIT REPORT")
    print("=" * W)

    print(f"\nMODELS MATCHING BEST-KNOWN RANK  ({len(hits_bk)} hits)")
    print("-" * W)
    if hits_bk:
        for gene_id, size, R, bk, ae in hits_bk:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (best_known={bk})")
    else:
        print("  (none)")

    hits_bk_only = [(g, s, R, bk, ae) for (g, s, R, bk, ae) in hits_bk
                    if not any(g == g2 and s == s2 for g2, s2, _, _ in hits_ae)]

    print(f"\nMODELS BEATING BEST-KNOWN BUT NOT ALPHAEVOLVE  ({len(hits_bk_only)} hits)")
    print("-" * W)
    if hits_bk_only:
        for gene_id, size, R, bk, ae in hits_bk_only:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (best_known={bk}, alphaevolve={ae})")
    else:
        print("  (none)")

    print(f"\nMODELS MATCHING ALPHAEVOLVE RANK  ({len(hits_ae)} hits)")
    print("-" * W)
    if hits_ae:
        for gene_id, size, R, ae in hits_ae:
            print(f"  {gene_id:<35} size={size:<14} R={R}  (alphaevolve={ae})")
    else:
        print("  (none)")

    print("=" * W)


if __name__ == "__main__":
    main()