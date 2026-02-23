import argparse
import os
from pathlib import Path as p
from os.path import join as pj

# --- Utilities ---
def create_save_dir(save_root: str) -> str:
    save_root_path = p(save_root)
    if not save_root_path.exists():
        save_root_path.mkdir(exist_ok=True, parents=True)

    n = []
    for exp_dir in save_root_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("exp"):
            try:
                n.append(int(exp_dir.name[3:]))
            except: pass

    if len(n) == 0:
        return pj(save_root, "exp1")
    else:
        return pj(save_root, f"exp{sorted(n)[-1] + 1}")

def resolve_exp_dir(args):
    if args.save_dir is not None:
        exp_dir = p(args.save_dir).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    return p(create_save_dir(args.save_root)).resolve()

# --- Core Logic ---
# --OPTION--
def generate_verilog_code(N: int) -> str:
    input_width = N
    output_width = 2 * N
    
    verilog = f"""
module multiplier_{N} (
    input [{input_width-1}:0] a,
    input [{input_width-1}:0] b,
    output [{output_width-1}:0] p
);
    // Baseline implementation
    assign p = a * b;
endmodule
    """
    return verilog
# --OPTION--

# --- Configuration ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8, help="Bit width")
    parser.add_argument("--save_root", type=str, default="trained", help="root directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Explicit experiment directory")
    return parser.parse_args()

# --- Main Execution ---
def main():
    args = get_args()
    exp_dir = resolve_exp_dir(args)
    
    print(f"Generating {args.N}-bit Multiplier")

    code = generate_verilog_code(args.N)
    
    design_path = pj(exp_dir, "design.v")
    with open(design_path, "w") as f:
        f.write(code)
        
    print(f"Design saved to: {design_path}")

if __name__ == "__main__":
    main()