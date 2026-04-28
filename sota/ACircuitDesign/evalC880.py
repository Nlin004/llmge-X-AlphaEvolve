import os
import sys
import subprocess
import argparse
import importlib
import re
from pathlib import Path as p
from os.path import join as pj

REF_V = p(__file__).parent / "c880.v"
TOTAL_TEST_VECTORS = 502
TOTAL_OUTPUT_BITS = TOTAL_TEST_VECTORS * 26

def write_testbench(run_dir):
    """Golden-model testbench: drives both c880 (reference) and c880_impl (DUT),
    compares all 26 outputs across 500 random input vectors."""
    tb_code = """
module tb;
    // 60 primary inputs
    reg N1, N8, N13, N17, N26, N29, N36, N42, N51, N55;
    reg N59, N68, N72, N73, N74, N75, N80, N85, N86, N87;
    reg N88, N89, N90, N91, N96, N101, N106, N111, N116, N121;
    reg N126, N130, N135, N138, N143, N146, N149, N152, N153, N156;
    reg N159, N165, N171, N177, N183, N189, N195, N201, N207, N210;
    reg N219, N228, N237, N246, N255, N259, N260, N261, N267, N268;

    // Reference outputs
    wire r388, r389, r390, r391, r418, r419, r420, r421, r422, r423;
    wire r446, r447, r448, r449, r450, r767, r768, r850;
    wire r863, r864, r865, r866, r874, r878, r879, r880;

    // DUT outputs
    wire d388, d389, d390, d391, d418, d419, d420, d421, d422, d423;
    wire d446, d447, d448, d449, d450, d767, d768, d850;
    wire d863, d864, d865, d866, d874, d878, d879, d880;

    // Packed output buses for single comparison
    wire [25:0] ref_out = {r388,r389,r390,r391,r418,r419,r420,r421,r422,r423,
                           r446,r447,r448,r449,r450,r767,r768,r850,
                           r863,r864,r865,r866,r874,r878,r879,r880};
    wire [25:0] dut_out = {d388,d389,d390,d391,d418,d419,d420,d421,d422,d423,
                           d446,d447,d448,d449,d450,d767,d768,d850,
                           d863,d864,d865,d866,d874,d878,d879,d880};

    integer i, j, vector_errors, bit_errors;
    reg [31:0] rnd1, rnd2;

    // Reference (golden model)
    c880 ref (
        .N1(N1),.N8(N8),.N13(N13),.N17(N17),.N26(N26),.N29(N29),.N36(N36),
        .N42(N42),.N51(N51),.N55(N55),.N59(N59),.N68(N68),.N72(N72),.N73(N73),
        .N74(N74),.N75(N75),.N80(N80),.N85(N85),.N86(N86),.N87(N87),.N88(N88),
        .N89(N89),.N90(N90),.N91(N91),.N96(N96),.N101(N101),.N106(N106),
        .N111(N111),.N116(N116),.N121(N121),.N126(N126),.N130(N130),.N135(N135),
        .N138(N138),.N143(N143),.N146(N146),.N149(N149),.N152(N152),.N153(N153),
        .N156(N156),.N159(N159),.N165(N165),.N171(N171),.N177(N177),.N183(N183),
        .N189(N189),.N195(N195),.N201(N201),.N207(N207),.N210(N210),.N219(N219),
        .N228(N228),.N237(N237),.N246(N246),.N255(N255),.N259(N259),.N260(N260),
        .N261(N261),.N267(N267),.N268(N268),
        .N388(r388),.N389(r389),.N390(r390),.N391(r391),
        .N418(r418),.N419(r419),.N420(r420),.N421(r421),.N422(r422),.N423(r423),
        .N446(r446),.N447(r447),.N448(r448),.N449(r449),.N450(r450),
        .N767(r767),.N768(r768),.N850(r850),
        .N863(r863),.N864(r864),.N865(r865),.N866(r866),
        .N874(r874),.N878(r878),.N879(r879),.N880(r880)
    );

    // DUT (LLM-generated candidate)
    c880_impl dut (
        .N1(N1),.N8(N8),.N13(N13),.N17(N17),.N26(N26),.N29(N29),.N36(N36),
        .N42(N42),.N51(N51),.N55(N55),.N59(N59),.N68(N68),.N72(N72),.N73(N73),
        .N74(N74),.N75(N75),.N80(N80),.N85(N85),.N86(N86),.N87(N87),.N88(N88),
        .N89(N89),.N90(N90),.N91(N91),.N96(N96),.N101(N101),.N106(N106),
        .N111(N111),.N116(N116),.N121(N121),.N126(N126),.N130(N130),.N135(N135),
        .N138(N138),.N143(N143),.N146(N146),.N149(N149),.N152(N152),.N153(N153),
        .N156(N156),.N159(N159),.N165(N165),.N171(N171),.N177(N177),.N183(N183),
        .N189(N189),.N195(N195),.N201(N201),.N207(N207),.N210(N210),.N219(N219),
        .N228(N228),.N237(N237),.N246(N246),.N255(N255),.N259(N259),.N260(N260),
        .N261(N261),.N267(N267),.N268(N268),
        .N388(d388),.N389(d389),.N390(d390),.N391(d391),
        .N418(d418),.N419(d419),.N420(d420),.N421(d421),.N422(d422),.N423(d423),
        .N446(d446),.N447(d447),.N448(d448),.N449(d449),.N450(d450),
        .N767(d767),.N768(d768),.N850(d850),
        .N863(d863),.N864(d864),.N865(d865),.N866(d866),
        .N874(d874),.N878(d878),.N879(d879),.N880(d880)
    );

    initial begin
        vector_errors = 0;
        bit_errors = 0;

        // Corner case: all zeros
        {N268,N267,N261,N260,N259,N255,N246,N237,N228,N219,N210,N207,
         N201,N195,N189,N183,N177,N171,N165,N159,N156,N153,N152,N149,
         N146,N143,N138,N135,N130,N126,N121,N116} = 32'h0;
        {N111,N106,N101,N96,N91,N90,N89,N88,N87,N86,N85,N80,N75,N74,
         N73,N72,N68,N59,N55,N51,N42,N36,N29,N26,N17,N13,N8,N1} = 32'h0;
        #10;
        if (ref_out !== dut_out) begin
            vector_errors = vector_errors + 1;
            for (j = 0; j < 26; j = j + 1)
                if (ref_out[j] !== dut_out[j]) bit_errors = bit_errors + 1;
        end

        // Corner case: all ones
        {N268,N267,N261,N260,N259,N255,N246,N237,N228,N219,N210,N207,
         N201,N195,N189,N183,N177,N171,N165,N159,N156,N153,N152,N149,
         N146,N143,N138,N135,N130,N126,N121,N116} = 32'hFFFFFFFF;
        {N111,N106,N101,N96,N91,N90,N89,N88,N87,N86,N85,N80,N75,N74,
         N73,N72,N68,N59,N55,N51,N42,N36,N29,N26,N17,N13,N8,N1} = 32'hFFFFFFFF;
        #10;
        if (ref_out !== dut_out) begin
            vector_errors = vector_errors + 1;
            for (j = 0; j < 26; j = j + 1)
                if (ref_out[j] !== dut_out[j]) bit_errors = bit_errors + 1;
        end

        // 500 random vectors
        for (i = 0; i < 500; i = i + 1) begin
            rnd1 = $random; rnd2 = $random;
            {N268,N267,N261,N260,N259,N255,N246,N237,N228,N219,N210,N207,
             N201,N195,N189,N183,N177,N171,N165,N159,N156,N153,N152,N149,
             N146,N143,N138,N135,N130,N126,N121,N116} = rnd1;
            {N111,N106,N101,N96,N91,N90,N89,N88,N87,N86,N85,N80,N75,N74,
             N73,N72,N68,N59,N55,N51,N42,N36,N29,N26,N17,N13,N8,N1} = rnd2;
            #10;
            if (ref_out !== dut_out) begin
                vector_errors = vector_errors + 1;
                for (j = 0; j < 26; j = j + 1)
                    if (ref_out[j] !== dut_out[j]) bit_errors = bit_errors + 1;
            end
        end

        if (vector_errors == 0) $display("RESULT: PASS");
        else $display("RESULT: FAIL (%0d vector errors, %0d bit errors)", vector_errors, bit_errors);
        $display("BIT_ERRORS: %0d / %0d", bit_errors, __TOTAL_OUTPUT_BITS__);
        $finish;
    end
endmodule
"""
    tb_path = pj(run_dir, "tb.v")
    tb_code = tb_code.replace("__TOTAL_OUTPUT_BITS__", str(TOTAL_OUTPUT_BITS))
    with open(tb_path, "w") as f:
        f.write(tb_code.strip())
    return tb_path

def estimate_ppa(verilog_file):
    """Heuristic PPA proxy for structural/behavioral Verilog without Yosys."""
    with open(verilog_file, 'r') as f:
        content = f.read()

    # Area: count boolean operators — each is one gate-like operation
    and_ops  = len(re.findall(r'&', content))
    or_ops   = len(re.findall(r'\|', content))
    not_ops  = len(re.findall(r'~', content))
    area = and_ops + or_ops + not_ops

    # Delay: count assign statements — each is one combinational level;
    # fewer assigns means a shallower or more merged dependency chain
    delay = len(re.findall(r'\bassign\b', content))
    if delay == 0:
        delay = 10  # baseline for fully flattened logic

    return area, delay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="SeedModel1")
    parser.add_argument('--save_dir', type=str, default="trained")
    parser.add_argument('--variant_dir', type=str, default='models')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    gene_id = args.model.split('model_')[1] if 'model_' in args.model else 'seed'
    run_dir = p(args.save_dir).resolve() / f"{gene_id}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    # 1. Run the LLM-generated Python script to produce design.v
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)
    sys.argv = [sys.argv[0], "--save_dir", str(run_dir)]
    model_module.main()

    # 2. Verify design.v was produced
    design_file = run_dir / "design.v"
    if not design_file.exists():
        print("ERROR: design.v not generated.")
        sys.exit(1)

    # 3. Simulate: compile design.v + reference c880.v + testbench
    tb_file = write_testbench(run_dir)
    sim_out = pj(run_dir, "sim.out")

    fitness_correctness = 1.0  # 1.0 = failed
    area, delay = 9999, 9999

    try:
        subprocess.check_output(
            ["iverilog", "-o", sim_out, str(design_file), str(REF_V), str(tb_file)],
            stderr=subprocess.STDOUT
        )
        res = subprocess.check_output(["vvp", sim_out]).decode()

        bit_error_match = re.search(r"BIT_ERRORS:\s*(\d+)\s*/\s*(\d+)", res)
        if bit_error_match:
            bit_errors = int(bit_error_match.group(1))
            total_bits = int(bit_error_match.group(2))
            fitness_correctness = bit_errors / total_bits if total_bits else 1.0
            area, delay = estimate_ppa(design_file)
        if "RESULT: PASS" in res:
            print(f"[SUCCESS] Correct logic! Area: {area}, Delay: {delay}")
        else:
            print(
                "[FAILED] Logic mismatch vs reference. "
                f"Correctness error rate: {fitness_correctness:.6f}. Area: {area}, Delay: {delay}\n{res}"
            )

    except FileNotFoundError as e:
        print(f"[SYNTAX ERROR] Required Verilog tool not found:\n{e}")
    except subprocess.CalledProcessError as e:
        print(f"[SYNTAX ERROR] iverilog failed:\n{e.output.decode()}")

    # 4. Write multi-objective fitness (correctness, area, delay)
    results_text = f"{fitness_correctness:.6f}, {area}, {delay}"

    res_file = (p(__file__).parent / 'results' / f'{gene_id}_results.txt').resolve()
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, 'w') as f:
        f.write(results_text)

    print(f"Metrics saved: {results_text}")
    print("Job Done")

if __name__ == '__main__':
    main()
