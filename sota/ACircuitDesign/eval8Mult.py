import os
import sys
import subprocess
import argparse
import importlib
import re
from pathlib import Path as p
from os.path import join as pj

def write_testbench(run_dir):
    """Writes a rigid testbench to verify the 8-bit multiplier."""
    tb_code = """
module tb;
    reg [7:0] a, b;
    wire [15:0] p;
    integer i, errors;
    
    mult8 uut (.a(a), .b(b), .p(p));

    initial begin
        errors = 0;
        // Test Corner Cases
        a = 8'h00; b = 8'h00; #10; if (p !== 16'h0000) errors = errors + 1;
        a = 8'hFF; b = 8'hFF; #10; if (p !== 16'hFE01) errors = errors + 1;
        a = 8'hFF; b = 8'h01; #10; if (p !== 16'h00FF) errors = errors + 1;
        
        // Test Random Vectors
        for (i = 0; i < 200; i = i + 1) begin
            a = $random; b = $random; #10;
            if (p !== (a * b)) begin
                $display("FAIL: %d * %d = %d (Expected %d)", a, b, p, a*b);
                errors = errors + 1;
            end
        end
        
        if (errors == 0) $display("RESULT: PASS");
        else $display("RESULT: FAIL (%d errors)", errors);
        $finish;
    end
endmodule
"""
    tb_path = pj(run_dir, "tb.v")
    with open(tb_path, "w") as f:
        f.write(tb_code.strip())
    return tb_path

def estimate_ppa(verilog_file):
    """Heuristic proxy for Area and Delay without needing Yosys."""
    with open(verilog_file, 'r') as f:
        content = f.read()
    
    # Area Proxy: Count additions and logic operations
    adders = len(re.findall(r'\+', content))
    logic_gates = len(re.findall(r'\b(and|or|xor)\b', content))
    area = (adders * 10) + logic_gates
    
    # Delay Proxy: Count the maximum depth of sequential variable assignments
    # A ripple chain (sum1 = pp0+pp1; sum2 = sum1+pp2) is deeply nested.
    delay = len(re.findall(r'sum\d+', content)) * 5 
    if delay == 0: delay = 10 # Baseline for flattened logic
    
    return area, delay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="SeedModel1")
    parser.add_argument('--save_dir', type=str, default="trained")
    parser.add_argument('--variant_dir', type=str, default='models')
    args = parser.parse_args()
    
    gene_id = args.model.split('model_')[1] if 'model_' in args.model else 'seed'
    run_dir = p(args.save_dir).resolve() / f"{gene_id}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run the LLM's Generated Python Script
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)
    sys.argv = [sys.argv[0], "--save_dir", str(run_dir)]
    model_module.main()

    # 2. Verify Output Exists
    design_file = run_dir / "design.v"
    if not design_file.exists():
        print("ERROR: design.v not generated.")
        sys.exit(1)

    # 3. Simulate Correctness
    tb_file = write_testbench(run_dir)
    sim_out = pj(run_dir, "sim.out")
    
    fitness_correctness = 1.0 # 1.0 = 100% Error (Fail)
    area, delay = 9999, 9999
    
    try:
        # Compile
        subprocess.check_output(["iverilog", "-o", sim_out, str(design_file), str(tb_file)], stderr=subprocess.STDOUT)
        # Run
        res = subprocess.check_output(["vvp", sim_out]).decode()
        
        if "RESULT: PASS" in res:
            fitness_correctness = 0.0 # 0% Error (Perfect)
            area, delay = estimate_ppa(design_file)
            print(f"[SUCCESS] Correct logic! Area: {area}, Delay: {delay}")
        else:
            print(f"[FAILED] Logic error in circuit.\n{res}")
            
    except subprocess.CalledProcessError as e:
        print(f"[SYNTAX ERROR] iverilog failed:\n{e.output.decode()}")

    # 4. Save Multi-Objective Fitness (Correctness, Area, Delay)
    results_text = f"{fitness_correctness:.1f}, {area}, {delay}"
    
    res_file = p(f'results/{gene_id}_results.txt').resolve()
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, 'w') as f:
        f.write(results_text)
        
    print(f"Metrics saved: {results_text}")

if __name__ == '__main__':
    main()