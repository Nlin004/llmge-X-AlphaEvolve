import argparse
import os
from pathlib import Path as p
from os.path import join as pj

def create_save_dir(save_root: str) -> str:
    save_root_path = p(save_root)
    save_root_path.mkdir(exist_ok=True, parents=True)
    n = [int(d.name[3:]) for d in save_root_path.iterdir() if d.is_dir() and d.name.startswith("exp") and d.name[3:].isdigit()]
    return pj(save_root, f"exp{sorted(n)[-1] + 1}" if n else "exp1")

# --OPTION--
def generate_seed_verilog() -> str:
    """
    GENERATION 0: Combinational Shift-and-Add Multiplier.
    Functionally correct, but High Area and High Delay.
    The LLM will mutate this block in future generations.
    """
    return """
module mult8 (
    input  [7:0] a,
    input  [7:0] b,
    output [15:0] p
);

    // Generate Partial Products
    wire [15:0] pp0 = b[0] ? {8'b0, a}       : 16'b0;
    wire [15:0] pp1 = b[1] ? {7'b0, a, 1'b0} : 16'b0;
    wire [15:0] pp2 = b[2] ? {6'b0, a, 2'b0} : 16'b0;
    wire [15:0] pp3 = b[3] ? {5'b0, a, 3'b0} : 16'b0;
    wire [15:0] pp4 = b[4] ? {4'b0, a, 4'b0} : 16'b0;
    wire [15:0] pp5 = b[5] ? {3'b0, a, 5'b0} : 16'b0;
    wire [15:0] pp6 = b[6] ? {2'b0, a, 6'b0} : 16'b0;
    wire [15:0] pp7 = b[7] ? {1'b0, a, 7'b0} : 16'b0;

    // Linear Reduction Tree (Terrible for Delay!)
    wire [15:0] sum1 = pp0 + pp1;
    wire [15:0] sum2 = sum1 + pp2;
    wire [15:0] sum3 = sum2 + pp3;
    wire [15:0] sum4 = sum3 + pp4;
    wire [15:0] sum5 = sum4 + pp5;
    wire [15:0] sum6 = sum5 + pp6;
    
    assign p = sum6 + pp7;

endmodule
"""
# --OPTION--

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    exp_dir = p(args.save_dir).resolve() if args.save_dir else p(create_save_dir("trained")).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    design_path = pj(exp_dir, "design.v")
    with open(design_path, "w") as f:
        f.write(generate_seed_verilog().strip())
        
    print(f"Seed Multiplier generated at: {design_path}")

if __name__ == "__main__":
    main()