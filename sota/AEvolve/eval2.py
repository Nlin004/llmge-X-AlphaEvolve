import argparse
import importlib
import os
from pathlib import Path as p
from os.path import join as pj
import sys
import numpy as np
import random


def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)    
    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    return save_dir


def verify_decomposition(m, n, p, decomposition, num_tests=20):
    """Verify decomposition correctness."""
    if decomposition is None:
        return False, float('inf'), float('inf')
    
    max_error = 0.0
    all_correct = True
    
    for _ in range(num_tests):
        A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        B = np.random.randn(n, p) + 1j * np.random.randn(n, p)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_flat = np.zeros(m * p, dtype=np.complex128)
        
        for u, v, w in decomposition:
            Au = np.dot(u, A_flat)
            Bv = np.dot(v, B_flat)
            scalar = Au * Bv
            C_flat += scalar * w
        
        C_result = C_flat.reshape(m, p)
        error = np.max(np.abs(C_result - C_expected))
        max_error = max(max_error, error)
        
        if error > 1e-6:
            all_correct = False
    
    num_multiplications = len(decomposition)
    return all_correct, max_error, num_multiplications


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model", help="model file")
    parser.add_argument('--save_dir', type=str, default="trained", help="path where results will be saved")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    parser.add_argument('--variant_dir', type=str, default='models', help="directory where models are written by LLM-GE")
    parser.add_argument('--m', type=int, default=2, help="rows in first matrix")
    parser.add_argument('--n', type=int, default=2, help="cols in first / rows in second")
    parser.add_argument('--p', type=int, default=2, help="cols in second matrix")
    parser.add_argument('--target_rank', type=int, default=7, help="target multiplications")
    return parser.parse_args()


if __name__ == '__main__':
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)
    args = get_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)
    
    try:
        gene_id = args.model.split('model_')[1]
    except:
        gene_id = 'seed'
    
    save_dir = f'{args.save_dir}/{gene_id}' 
    create_save_dir(save_dir)
    
    print("="*120)
    print(f"EVALUATING: {args.m}×{args.n} × {args.n}×{args.p} matrix multiplication")
    print(f"Gene ID: {gene_id}")
    print("="*120)
    
    model = model_module.Model(m=args.m, n=args.n, p=args.p, target_rank=args.target_rank)
    model.fit(num_steps=2000, learning_rate=0.01, seed=args.random_seed)
    decomposition = model.get_decomposition()
    
    print("\nVerifying correctness...")
    is_correct, max_error, num_multiplications = verify_decomposition(
        args.m, args.n, args.p, decomposition, num_tests=20
    )
    
    standard_mults = args.m * args.n * args.p
    
    if not is_correct:
        fp_count = 1000000
        fn_count = int(max_error * 1000)
    else:
        fp_count = num_multiplications
        fn_count = 0
    
    print("\n" + "="*120)
    print("RESULTS:")
    print("="*120)
    print(f"Rank: {num_multiplications}, Standard: {standard_mults}, Correct: {is_correct}, Error: {max_error:.2e}")
    print(f"fp_count (num_mults): {fp_count}, fn_count: {fn_count}")
    print("="*120)
    
    print(fp_count, fn_count)
    
    results_text = f"{fp_count},{fn_count}"
    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    dir_path = os.path.dirname(filename)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(filename, 'w') as file:
        file.write(results_text)
    
    print(f"\nResults written to {filename}")
    print('='*120)
    print('job done')
    print('='*120)