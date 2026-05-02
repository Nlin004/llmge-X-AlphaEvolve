import argparse
import ast
import sys
sys.path.append("src")
import re
import os
import glob
import time
import numpy as np
import transformers
from torch import bfloat16
from utils.privit import *
from cfg.constants import *
from utils.print_utils import box_print

from typing import Optional
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
#from transformers import AutoTokenizer
from google import genai
from google.genai import types


def safe_prompt_format(template_text, *values):
    """
    Fill only bare '{}' prompt placeholders. Literal braces in prompt examples
    must not be interpreted as Python str.format fields.
    """
    formatted = template_text
    for value in values:
        if "{}" not in formatted:
            raise ValueError("Prompt template has fewer '{}' placeholders than expected.")
        formatted = formatted.replace("{}", str(value), 1)
    return formatted


C880_INPUT_PORTS = (
    "N1", "N8", "N13", "N17", "N26", "N29", "N36", "N42", "N51", "N55",
    "N59", "N68", "N72", "N73", "N74", "N75", "N80", "N85", "N86", "N87",
    "N88", "N89", "N90", "N91", "N96", "N101", "N106", "N111", "N116", "N121",
    "N126", "N130", "N135", "N138", "N143", "N146", "N149", "N152", "N153", "N156",
    "N159", "N165", "N171", "N177", "N183", "N189", "N195", "N201", "N207", "N210",
    "N219", "N228", "N237", "N246", "N255", "N259", "N260", "N261", "N267", "N268",
)

C880_OUTPUT_PORTS = (
    "N388", "N389", "N390", "N391", "N418", "N419", "N420", "N421", "N422", "N423",
    "N446", "N447", "N448", "N449", "N450", "N767", "N768", "N850", "N863", "N864",
    "N865", "N866", "N874", "N878", "N879", "N880",
)

C880_PORTS = C880_INPUT_PORTS + C880_OUTPUT_PORTS


def get_template_root():
    return globals().get("TEMPLATE_DIR", os.path.join(ROOT_DIR, "templates_AE"))


def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()


def _top_level_defs(code_text):
    """Return the names of top-level functions/classes defined in a code block."""
    try:
        tree = ast.parse(code_text)
    except SyntaxError as exc:
        raise ValueError(f"Candidate block is not valid Python: {exc}") from exc

    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return tree, names


def _extract_c880_module_header(code_text):
    match = re.search(r"module\s+c880_impl\s*\((.*?)\)\s*;", code_text, flags=re.DOTALL)
    if not match:
        raise ValueError("Candidate does not contain a parseable c880_impl module header.")
    return match.group(1)


def validate_c880_verilog_contract(code_text):
    """
    Enforce the exact scalar C880 DUT interface expected by evalC880.py.
    This rejects LLM variants that replace the 60/26 named ports with bus
    aliases like A/Y, or otherwise mutate the module contract.
    """
    header = _extract_c880_module_header(code_text)

    if re.search(r"\b(?:input|output)\s*\[", header):
        raise ValueError(
            "c880_impl must keep the scalar named-port interface; bus ports are not allowed."
        )

    header_names = [
        token for token in re.findall(r"\b[A-Za-z_]\w*\b", header)
        if token not in {"input", "output"}
    ]

    if header_names != list(C880_PORTS):
        missing = [name for name in C880_PORTS if name not in header_names]
        extras = [name for name in header_names if name not in C880_PORTS]

        details = []
        if missing:
            details.append(f"missing ports: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if extras:
            details.append(f"unexpected ports: {extras[:8]}{'...' if len(extras) > 8 else ''}")
        if not details:
            details.append("port order or direction no longer matches the required C880 interface")

        raise ValueError("Invalid c880_impl port interface: " + "; ".join(details))

    return True

def validate_c880_signal_closure(code_text):
    """Reject C880 variants with duplicate, undeclared, or undriven signals."""
    if "module c880_impl" not in code_text:
        return True

    declared = set(C880_PORTS)
    for wire_group in re.findall(r"\bwire\b\s+([^;]+);", code_text, flags=re.DOTALL):
        declared.update(re.findall(r"\bN\d+\b", wire_group))

    lhs_signals = re.findall(r"\bassign\s+(N\d+)\s*=", code_text)
    assigned = set(lhs_signals)
    duplicates = sorted({signal for signal in lhs_signals if lhs_signals.count(signal) > 1}, key=lambda name: int(name[1:]))
    if duplicates:
        raise ValueError(f"Candidate assigns the same C880 signal more than once: {duplicates[:12]}")

    unknown_lhs = sorted(assigned - declared, key=lambda name: int(name[1:]))
    if unknown_lhs:
        raise ValueError(f"Candidate assigns undeclared C880 signals: {unknown_lhs[:12]}")

    referenced = set()
    for rhs in re.findall(r"\bassign\s+N\d+\s*=\s*([^;]+);", code_text, flags=re.DOTALL):
        referenced.update(re.findall(r"\bN\d+\b", rhs))

    unknown_refs = sorted(referenced - declared, key=lambda name: int(name[1:]))
    if unknown_refs:
        raise ValueError(f"Candidate references undeclared C880 signals: {unknown_refs[:12]}")

    undriven_refs = sorted(
        referenced - set(C880_INPUT_PORTS) - assigned,
        key=lambda name: int(name[1:]),
    )
    if undriven_refs:
        raise ValueError(f"Candidate references undriven C880 signals: {undriven_refs[:12]}")

    undriven_outputs = sorted(
        set(C880_OUTPUT_PORTS) - assigned,
        key=lambda name: int(name[1:]),
    )
    if undriven_outputs:
        raise ValueError(f"Candidate leaves output ports undriven: {undriven_outputs[:12]}")

    return True


def validate_candidate_block(candidate_code, base_code):
    """
    Validate that a mutated block still matches the structural contract of the
    block it replaced. This prevents semantically unrelated Python from being
    stitched into the seed file.
    """
    candidate_tree, candidate_names = _top_level_defs(candidate_code)
    _, base_names = _top_level_defs(base_code)

    missing_names = [name for name in base_names if name not in candidate_names]
    if missing_names:
        raise ValueError(f"Candidate block is missing required definitions: {missing_names}")

    if "generate_seed_verilog" in base_names:
        if "module c880_impl" not in candidate_code:
            raise ValueError("Candidate generate_seed_verilog block no longer contains the c880_impl module.")
        validate_c880_verilog_contract(candidate_code)
        validate_c880_signal_closure(candidate_code)

        # The mutation target for C880 should remain a pure definition block,
        # not an executable script that runs arbitrary code during import.
        disallowed_nodes = (
            ast.For, ast.While, ast.If, ast.With, ast.Try, ast.Match,
            ast.Expr,
        )
        for node in candidate_tree.body:
            if isinstance(node, disallowed_nodes):
                # Allow docstring-like constant expressions only.
                if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                    continue
                raise ValueError("Candidate block contains top-level executable statements unrelated to the Verilog generator.")

    if "main" in base_names:
        required_strings = ["--save_dir", "design.v"]
        for needle in required_strings:
            if needle not in candidate_code:
                raise ValueError(f"Candidate main block is missing required token: {needle}")

    return True


def validate_augmented_file(full_code_text):
    """Validate the stitched candidate module before it is written to disk."""
    try:
        tree = ast.parse(full_code_text)
    except SyntaxError as exc:
        raise ValueError(f"Full candidate file is not valid Python: {exc}") from exc

    defined_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    required_names = {"create_save_dir", "generate_seed_verilog", "main"}
    missing = sorted(required_names - defined_names)
    if missing:
        raise ValueError(f"Full candidate file is missing required definitions: {missing}")

    if "module c880_impl" not in full_code_text:
        raise ValueError("Full candidate file is missing the c880_impl Verilog payload.")
    validate_c880_verilog_contract(full_code_text)
    validate_c880_signal_closure(full_code_text)

    return True

def _env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

def _llm_request_timeout():
    value = os.getenv("LLM_HTTP_TIMEOUT")
    if value is None or value.strip().lower() in {"", "0", "none", "false", "off"}:
        return None
    return int(value)

def is_valid_python(code_text):
    """Return True when a candidate can at least be imported by Python."""
    try:
        ast.parse(code_text)
    except SyntaxError:
        return False
    return True

def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    if not code_from_llm:
        raise ValueError("No code received from the LLM.")

    fenced_blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", code_from_llm, flags=re.DOTALL)
    if fenced_blocks:
        # Prefer the largest fenced block because some models emit a short
        # example followed by the real full candidate.
        return max(fenced_blocks, key=len).strip()

    code_generator = None
    # Select Correct LLM
    if LLM_MODEL == 'mixtral' or LLM_MODEL == 'llama3.3':
        code_generator = submit_mixtral_local
    elif LLM_MODEL == 'llama3':
        code_generator = submit_llama3_hf
    elif LLM_MODEL == 'gemini':
        code_generator = submit_gemini_api
    elif LLM_MODEL == 'deepseek':
        code_generator = submit_deepseek_local
        # code_checker_prompt = os.path.join(ROOT_DIR, 'templates/FixedPrompts/validation/code_validation_prompt.txt')
        # model_varaint_code = ""
        # if "```" in code_from_llm:
        #     model_varaint_code = '\n'.join(code_from_llm.split("```")[1].strip().split("\n")[1:])
        # else:
        #     model_varaint_code = None
        # if model_varaint_code:
        #     box_print("VALIDATING LLM CODE", print_bbox_len=60, new_line_end=False)
        #     template_text = ""
        #     with open(code_checker_prompt, 'r') as file:
        #         template_text = file.read()
        #     prompt = template_text.format(model_varaint_code.strip())
        #     print(prompt)
        #     verified_code = code_generator(prompt, top_p=0.15, temperature=0.1) 
        #     print(verified_code)
        #     return '\n'.join(verified_code.strip().split("```")[1].split('\n')[1:])
    stripped = code_from_llm.strip()
    if stripped.startswith(("def ", "class ", "from ", "import ", "module ", "```python", "```verilog")):
        return stripped

    raise ValueError("LLM response did not include a recognizable code block.")


def _is_c880_verilog_block(code_text):
    return "def generate_seed_verilog" in code_text and "module c880_impl" in code_text

def _assign_line_indices(code_text):
    lines = code_text.splitlines()
    return [
        idx for idx, line in enumerate(lines)
        if line.strip().startswith("assign ") and line.rstrip().endswith(";")
    ]

def _build_c880_partial_mutation_prompt(base_code, source_prompt):
    """Create a compact LLM prompt that mutates only a small assign cluster."""
    if not _env_flag("LLM_C880_PARTIAL_MUTATION", True):
        return None
    if not _is_c880_verilog_block(base_code):
        return None

    assign_indices = _assign_line_indices(base_code)
    if not assign_indices:
        return None

    window_size = max(1, _env_int("C880_LLM_ASSIGN_WINDOW", 72))
    window_size = min(window_size, len(assign_indices))
    start = np.random.randint(0, len(assign_indices) - window_size + 1)
    selected_indices = assign_indices[start:start + window_size]
    lines = base_code.splitlines()
    selected_assigns = "\n".join(lines[idx].strip() for idx in selected_indices)
    selected_lhs = [
        re.match(r"\s*assign\s+([A-Za-z_]\w*)\s*=", lines[idx]).group(1)
        for idx in selected_indices
    ]
    allowed_signals = sorted({
        token
        for token in re.findall(r"\b[A-Za-z_]\w*\b", selected_assigns)
        if token != "assign"
    }, key=lambda name: (not name.startswith("N"), name))
    selected_lhs_set = set(selected_lhs)
    outside_text = "\n".join(
        line for idx, line in enumerate(lines)
        if idx < selected_indices[0] or idx > selected_indices[-1]
    )
    externally_required_lhs = sorted(
        selected_lhs_set & set(re.findall(r"\bN\d+\b", outside_text)),
        key=lambda name: int(name[1:]),
    )

    prompt_hint = source_prompt.split("```python", 1)[0].strip()
    compact_prompt = f"""
You are mutating a small part of the ISCAS-85 C880 Verilog implementation for evolutionary search.

Keep this a local edit. Return between 1 and {len(selected_indices)} Verilog assign statements replacing the selected block below. You may remove redundant assignments or rewrite right-hand side expressions when doing so preserves correctness. Do not add wires, modules, ports, comments, prose, markdown, Python, always blocks, buses, clocks, or resets.

Hard signal-name rule:
- Use only these signal names: {", ".join(allowed_signals)}
- The only left-hand-side names you may assign are: {", ".join(selected_lhs)}
- These downstream-required left-hand-side names must still be assigned exactly once: {", ".join(externally_required_lhs) if externally_required_lhs else "(none)"}
- Do not invent adjacent-number signal names such as N320 when only N319 is shown.
- Do not assign the same left-hand-side name more than once.
- If you are uncertain about an assignment, copy that assignment unchanged.
Primary goal: preserve exact functional correctness. Secondary goal: reduce area and dependency depth by removing redundant assign statements or simplifying local Boolean expressions.

Original guidance:
{prompt_hint}

Assign statements to locally mutate:
```verilog
{selected_assigns}
```

Return only the replacement assign statements. Start with the first assign statement and end with the last semicolon.
""".strip()

    return compact_prompt, selected_indices

def _clean_c880_assign_replacements(
    llm_text,
    expected_lhs,
    allowed_identifiers=None,
    required_lhs=None,
):
    if not llm_text:
        raise ValueError("No assign replacements received from the LLM.")

    fenced_blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", llm_text, flags=re.DOTALL)
    candidate = max(fenced_blocks, key=len).strip() if fenced_blocks else llm_text.strip()
    assignments = re.findall(r"\bassign\s+([A-Za-z_]\w*)\s*=\s*[^;]+;", candidate, flags=re.DOTALL)
    assign_lines = re.findall(r"\bassign\s+[A-Za-z_]\w*\s*=\s*[^;]+;", candidate, flags=re.DOTALL)

    normalized_lines = ["    " + " ".join(line.split()) for line in assign_lines]
    if not normalized_lines:
        raise ValueError("Expected at least one assign replacement.")

    expected_lhs_set = set(expected_lhs)
    unexpected_lhs = sorted(set(assignments) - expected_lhs_set)
    if unexpected_lhs:
        raise ValueError(
            "Assign replacements used left-hand-side names outside the selected block: "
            f"{unexpected_lhs}."
        )
    duplicate_lhs = sorted({lhs for lhs in assignments if assignments.count(lhs) > 1})
    if duplicate_lhs:
        raise ValueError(f"Assign replacements duplicate left-hand-side names: {duplicate_lhs}.")

    if required_lhs is not None:
        missing_required = sorted(set(required_lhs) - set(assignments))
        if missing_required:
            raise ValueError(
                "Assign replacements removed signals still required downstream: "
                f"{missing_required}."
            )

    if allowed_identifiers is not None:
        allowed_identifiers = set(allowed_identifiers)
        for line in normalized_lines:
            identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", line))
            identifiers.discard("assign")
            unknown = sorted(identifiers - allowed_identifiers)
            if unknown:
                raise ValueError(
                    "Assign replacements introduced out-of-window or undefined signal names: "
                    f"{unknown}. Allowed local signals are {sorted(allowed_identifiers)}."
                )

    return normalized_lines

def _stitch_c880_assign_replacements(base_code, selected_indices, replacement_lines):
    lines = base_code.splitlines()
    lines = lines[:selected_indices[0]] + replacement_lines + lines[selected_indices[-1] + 1:]
    return "\n".join(lines) + ("\n" if base_code.endswith("\n") else "")


def local_c880_refactor_fallback(base_code):
    """Create a valid C880 variant when the LLM server cannot return text."""
    if "def generate_seed_verilog" not in base_code or "module c880_impl" not in base_code:
        return base_code

    replacements = [
        ("    assign N369 = ~N310;", "    assign N369 = N268;"),
        ("    assign N393 = ~N345;", "    assign N393 = N276;"),
        ("    assign N399 = ~N346;", "    assign N399 = N276;"),
        ("    assign N451 = ~N424;", "    assign N451 = N400;"),
        ("    assign N735 = ~N662;", "    assign N735 = N590;"),
        ("    assign N738 = ~N670;", "    assign N738 = N597;"),
        ("    assign N741 = ~N678;", "    assign N741 = N606;"),
        ("    assign N744 = ~N687;", "    assign N744 = N616;"),
        ("    assign N747 = ~N697;", "    assign N747 = N625;"),
        ("    assign N750 = ~N705;", "    assign N750 = N632;"),
        ("    assign N753 = ~N713;", "    assign N753 = N641;"),
        ("    assign N756 = ~N722;", "    assign N756 = N651;"),
        ("    assign N840 = ~N829;", "    assign N840 = N811;"),
        ("    assign N855 = ~N846;", "    assign N855 = N837;"),
        ("    assign N856 = ~N847;", "    assign N856 = N838;"),
        ("    assign N857 = ~N848;", "    assign N857 = N839;"),
        ("    assign N870 = ~N862;", "    assign N870 = N854;"),
        ("    assign N875 = ~N871;", "    assign N875 = N867;"),
        ("    assign N876 = ~N872;", "    assign N876 = N868;"),
        ("    assign N877 = ~N873;", "    assign N877 = N869;"),
    ]

    np.random.shuffle(replacements)
    max_changes = min(len(replacements), int(os.getenv("C880_LOCAL_FALLBACK_CHANGES", "4")))
    num_changes = np.random.randint(1, max_changes + 1)
    candidate = base_code
    applied = []

    for old, new in replacements:
        if old in candidate:
            candidate = candidate.replace(old, new, 1)
            applied.append(new.strip())
        if len(applied) >= num_changes:
            break

    if applied:
        print("Local C880 fallback applied:", "; ".join(applied), flush=True)
    return candidate

def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature,
                            inference_submission=False, allow_invalid_candidate=False):
    """Generates augmented code using Mixtral."""

    base_code = retrieve_base_code(augment_idx)
    partial_mutation = _build_c880_partial_mutation_prompt(base_code, txt2llm)
    if partial_mutation is not None:
        txt2llm, selected_assign_indices = partial_mutation
        base_lines = base_code.splitlines()
        selected_lhs = [
            re.match(r"\s*assign\s+([A-Za-z_]\w*)\s*=", base_lines[idx]).group(1)
            for idx in selected_assign_indices
        ]
        selected_identifiers = set()
        for idx in selected_assign_indices:
            selected_identifiers.update(re.findall(r"\b[A-Za-z_]\w*\b", base_lines[idx]))
        selected_identifiers.discard("assign")
        outside_text = "\n".join(
            line for idx, line in enumerate(base_lines)
            if idx < selected_assign_indices[0] or idx > selected_assign_indices[-1]
        )
        required_selected_lhs = set(selected_lhs) & set(re.findall(r"\bN\d+\b", outside_text))
        print(
            "Using compact C880 partial mutation prompt with "
            f"{len(selected_assign_indices)} assign statements.",
            flush=True,
        )
    else:
        selected_assign_indices = None
        selected_lhs = None
        selected_identifiers = None
        required_selected_lhs = None

    box_print("PROMPT TO LLM", print_bbox_len=60, new_line_end=False)

    print(txt2llm, flush=True) # if you don't Flush the buffer it won't print immediately | James Tip
    
    if inference_submission is False:
        llm_code_generator = submit_mixtral_local
        qc_func = llm_code_qc
    else:
        if LLM_MODEL == 'mixtral' or LLM_MODEL == 'llama3.3':
            llm_code_generator = submit_mixtral_local
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        elif LLM_MODEL == 'gemini':
            llm_code_generator = submit_gemini_api
        elif LLM_MODEL == 'deepseek':
            llm_code_generator = submit_deepseek_local
        qc_func = llm_code_qc_hf

    retries = 0
    max_retries = int(os.getenv("LLM_RETRIES", "1"))
    fallback_code = None
    fallback_error = None
    while retries < max_retries:
        if apply_quality_control and selected_assign_indices is None:
            llm_result = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
            if not llm_result:
                retries += 1
                print("Response Invalid: LLM server returned no output")
                time.sleep(5)
                continue

            code_from_llm, generate_text = llm_result
            if not code_from_llm:
                retries += 1
                print("Response Invalid: LLM server returned empty text")
                time.sleep(5)
                continue

            code_from_llm = qc_func(code_from_llm, base_code, generate_text)
        else:
            code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)

        print("Checking LLM Response")

        if not code_from_llm :
            retries += 1
            print("Response Invalid: LLM server returned no output")
            time.sleep(5)
            continue

        try:
            if selected_assign_indices is not None:
                replacement_lines = _clean_c880_assign_replacements(
                    code_from_llm,
                    selected_lhs,
                    allowed_identifiers=selected_identifiers,
                    required_lhs=required_selected_lhs,
                )
                cleaned_code = _stitch_c880_assign_replacements(
                    base_code,
                    selected_assign_indices,
                    replacement_lines,
                )
            else:
                cleaned_code = clean_code_from_llm(code_from_llm)
            if is_valid_python(cleaned_code):
                fallback_code = cleaned_code
            validate_candidate_block(cleaned_code, base_code)
        except ValueError as exc:
            fallback_error = exc
            retries += 1
            print(f"Response Invalid: {exc}")
            continue

        print("Response Valid")
        box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
        print(cleaned_code)
        return cleaned_code

    if allow_invalid_candidate and fallback_code is not None:
        print(
            "Returning syntactically valid candidate despite validation failure "
            f"so the evaluator can assign partial/worst-case fitness: {fallback_error}",
            flush=True,
        )
        return fallback_code

    if allow_invalid_candidate:
        fallback_code = local_c880_refactor_fallback(base_code)
        try:
            validate_candidate_block(fallback_code, base_code)
            print(
                "LLM did not return any usable text after retries; returning a local "
                "correctness-preserving C880 fallback.",
                flush=True,
            )
            return fallback_code
        except ValueError as exc:
            print(f"Local C880 fallback failed validation: {exc}", flush=True)

        print(
            "LLM did not return any usable text after retries; returning the original "
            "block unchanged so this gene can still be evaluated.",
            flush=True,
        )
        return base_code

    raise RuntimeError(f"Failed to get a valid response from the LLM after {max_retries} retries.")

def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(get_template_root(), 'llm_quality_control.txt')
    if not os.path.exists(template_path):
        return code_from_llm
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = safe_prompt_format(template_txt, code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    if generate_text is None:
        code_from_llm = submit_mixtral_local(
            prompt2llm,
            max_new_tokens=512,
            top_p=0.1,
            temperature=0.1,
            return_gen=False,
        )
    else:
        res = generate_text(prompt2llm) # clean txt
        code_from_llm = res[0]["generated_text"]
    code_from_llm = clean_code_from_llm(code_from_llm).strip()
    return code_from_llm

def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(get_template_root(), fname)
    if not os.path.exists(template_path):
        return code_from_llm
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = safe_prompt_format(template_txt, code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_local(prompt2llm, max_new_tokens=512, top_p=0.1, temperature=0.1,
                      return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm

def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    """
    This function submits a model prompt to mixtral through the HuggingFace Inference API

    Parameters
    ----------
    txt2mixtral : str
        Prompt that will be sent to mixtral
    max_new_tokens : int, optional
       A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
       Which mixtral variant to utilize for inference, by default "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """   
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = transformers.AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False, 
        task="text-generation",
        temperature=temperature, 
        top_p=top_p,  
        top_k=0, 
        max_new_tokens=max_new_tokens, 
        repetition_penalty=1.1,
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def get_llm_server_hostname():
    hostname = None
    hostname_file_path = HOSTNAME_DIR 
    with open(hostname_file_path, 'r') as f:
        hostname = f.readline().strip() 
    return hostname

def submit_mixtral_local(prompt, max_new_tokens=256, temperature=0.2, top_p=0.15, server_url=f"http://{os.getenv('SERVER_HOSTNAME', 'localhost')}:{PORT}/generate", return_gen=False):
    
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }

    headers = {"Content-Type": "application/json"}

    llm_hostname = get_llm_server_hostname()
    
    print(llm_hostname)

    server_url = f"http://{llm_hostname}:{PORT}/generate"
    
    try:
        response = requests.post(
            server_url,
            headers=headers,
            json=payload,
            timeout=_llm_request_timeout(),
        )
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, None
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def submit_deepseek_local(prompt, max_new_tokens=256, temperature=0.2, top_p=0.15, server_url=f"http://{get_llm_server_hostname()}:8000/generate", return_gen=False):
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }
    print(os.getenv("SERVER_HOSTNAME", "localhost"))

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(
            server_url,
            headers=headers,
            json=payload,
            timeout=_llm_request_timeout(),
        )
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, None
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def submit_llama3_hf(txt2llama, 
                     max_new_tokens=1024, 
                     top_p=0.15, 
                     temperature=0.1,                   
                     model_id="google/gemma-2-27b-it",
                     return_gen=False):
    """
    This function submits a model prompt to Llama3 through the HuggingFace Inference API

    Parameters
    ----------
    txt2llama : str
        Prompt that will be sent to Llama3
    max_new_tokens : int, optional
        A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
        Which Llama3 variant to utilize for inference, by default "meta-llama/Meta-Llama-3.1-70B-Instruct"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """    
    max_new_tokens = np.random.randint(900, 1300)
    
    os.environ['HF_API_KEY'] = "DONT_SCRAPE_ME" 
    huggingface_hub.login(new_session=False)
    
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [
        {
            "role": "user",
            "content": "Provide code in Python\n" + txt2llama,
        },
    ]

    tokenizer_converter = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer_converter.add_special_tokens({'pad_token': '[PAD]'})
    prompt = f"{instructions[0]['role']}: {instructions[0]['content']}\n"
    encoded_prompt = tokenizer_converter.encode(
        prompt, 
        return_tensors='pt', 
        padding=True, 
        truncation=True
    )
    results = client.text_generation(
        encoded_prompt, 
        max_new_tokens=max_new_tokens, 
        return_full_text=False, 
        temperature=temperature, 
        seed=101
    )
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_gemini_api(txt2gemini, **kwargs):
    """
    This function submits a model prompt to Gemini through its API

    Parameters
    ----------
    txt2gemini : str
        Prompt that will be sent to Gemini

    Returns
    -------
    str
        Model's output from inference
    """   
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[txt2gemini],
        
    )
    return response.text

def mutate_prompts(n=5):
    template_paths = glob.glob(os.path.join(get_template_root(), 'FixedPrompts', '*', '*.txt'))
    if len(template_paths) == 0:
        return
    sample_count = min(n, len(template_paths))
    templates = np.random.choice(template_paths, sample_count, replace=False)
    for i, template in enumerate(templates):
        path, filename = os.path.split(template)
        with open(template, 'r') as file:
            prompt_text = file.read()
        prompt_text = prompt_text.split("```")[0].strip()
        prompt = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
        temp = np.random.uniform(0.01, 0.4)
        if LLM_MODEL == 'mixtral' or LLM_MODEL == 'llama3.3':
            llm_code_generator = submit_mixtral_local
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        elif LLM_MODEL == 'gemini':
            llm_code_generator = submit_gemini_api
        elif LLM_MODEL == 'deepseek':
            llm_code_generator = submit_deepseek_local
        output = llm_code_generator(prompt, temperature=temp).strip()
        if "```" in output:
            output = output.split("```")[0]
        output = output + "\n```python\n{}\n```"
        with open(os.path.join(path, "mutant{}.txt".format(i)), 'w') as file:
            file.write(output)
