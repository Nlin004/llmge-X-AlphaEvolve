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

def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature, inference_submission=False):
    """Generates augmented code using Mixtral."""

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
        
    base_code = retrieve_base_code(augment_idx)
    retries = 0
    while retries < 3:
        if apply_quality_control:
            code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
            code_from_llm = qc_func(code_from_llm, base_code, generate_text)
        else:
            code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)

        print("Checking LLM Response")

        if not code_from_llm :
            retries += 1
            print("Response Invalid")
            continue

        try:
            cleaned_code = clean_code_from_llm(code_from_llm)
            validate_candidate_block(cleaned_code, base_code)
        except ValueError as exc:
            retries += 1
            print(f"Response Invalid: {exc}")
            continue

        print("Response Valid")
        box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
        print(cleaned_code)
        return cleaned_code

    raise RuntimeError("Failed to get a valid response from the LLM after 3 retries.")

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
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    if generate_text is None:
        code_from_llm = submit_mixtral_local(
            prompt2llm,
            max_new_tokens=1500,
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
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_local(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
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

def submit_mixtral_local(prompt, max_new_tokens=850, temperature=0.2, top_p=0.15, server_url=f"http://{os.getenv('SERVER_HOSTNAME', 'localhost')}:{PORT}/generate", return_gen=False):
    
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
        response = requests.post(server_url, headers=headers, json=payload, timeout=600)
        
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

def submit_deepseek_local(prompt, max_new_tokens=850, temperature=0.2, top_p=0.15, server_url=f"http://{get_llm_server_hostname()}:8000/generate", return_gen=False):
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }
    print(os.getenv("SERVER_HOSTNAME", "localhost"))

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, json=payload, timeout=600)
        
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
