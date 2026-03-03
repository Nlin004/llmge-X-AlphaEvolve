import argparse
import importlib
import os
import sys
import subprocess
import re
from pathlib import Path as p
from os.path import join as pj

def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)
    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("exp"):
            try:
                n.append(int(exp_dir.name[3:]))
            except: pass
    if len(n) == 0: return pj(save_root, "exp1")
    else: return pj(save_root, f"exp{sorted(n)[-1] + 1}")

# --- Verification Helpers ---

def write_testbench(N, run_dir):
    tb_code = f"""
module tb;
    reg [{N-1}:0] a, b;
    wire [{2*N-1}:0] p;
    integer i, errors = 0;
    
    multiplier_{N} uut (.a(a), .b(b), .p(p));

    initial begin
        for (i = 0; i < 50; i = i + 1) begin
            a = $random;
            b = $random;
            #10;
            if (p !== a * b) begin
                $display("FAIL: %d * %d = %d", a, b, p);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("RESULT: PASS");
        else $display("RESULT: FAIL");
        $finish;
    end
endmodule
    """
    tb_path = pj(run_dir, "tb.v")
    with open(tb_path, "w") as f:
        f.write(tb_code)
    return tb_path

def estimate_area(verilog_file):
    with open(verilog_file, 'r') as f:
        content = f.read()
    # Simple heuristic for gate count
    gates = len(re.findall(r'\b(and|or|xor|nand|nor|not)\b', content))
    assigns = len(re.findall(r'\bassign\b', content)) * 2
    operators = len(re.findall(r'[\*\+\-\&\|\^]', content)) * 5
    return gates + assigns + operators

# --- Main Evaluation ---

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="SeedModel1", help="model file")
    parser.add_argument('--save_dir', type=str, default="trained", help="path")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    parser.add_argument('--variant_dir', type=str, default='models', help="dir")
    parser.add_argument('--N', type=int, default=8, help="bit width")
    return parser.parse_args()

if __name__ == '__main__':
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)
    args = get_args()
    
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)
    
    try: gene_id = args.model.split('model_')[1]
    except: gene_id = 'seed'

    run_dir = p(args.save_dir).resolve() / f"{gene_id}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {args.N}-bit Multiplier | Gene: {gene_id}")

    # Run SeedModel
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--save_dir", str(run_dir), "--N", str(args.N)]
    model_module.main()
    sys.argv = old_argv

    design_file = run_dir / "design.v"
    if not design_file.exists():
        raise FileNotFoundError(f"No design.v found in {run_dir}")

    # Simulation
    tb_file = write_testbench(args.N, run_dir)
    sim_out = pj(run_dir, "sim.out")
    cmd_compile = ["iverilog", "-o", sim_out, str(design_file), str(tb_file)]
    
    compile_success = False
    try:
        subprocess.check_output(cmd_compile, stderr=subprocess.STDOUT)
        compile_success = True
    except subprocess.CalledProcessError:
        pass

    fitness_correctness = 1.0 # 0 is perfect, 1 is fail
    fitness_area = 99999

    if compile_success:
        try:
            res = subprocess.check_output(["vvp", sim_out]).decode()
            if "RESULT: PASS" in res:
                fitness_correctness = 0.0
            fitness_area = estimate_area(design_file)
        except: pass

    # Results
    results_text = f"{fitness_correctness:.4f}, {fitness_area}"
    
    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(results_text)

    print(f"Fitness: {results_text}")
    print(f"Results written to {filename}")