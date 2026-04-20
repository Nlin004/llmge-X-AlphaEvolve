import os
import sys
import numpy as np
import platform

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOTA_ROOT = os.path.join(ROOT_DIR, 'sota/ACircuitDesign')
SEED_NETWORK = os.path.join(SOTA_ROOT, 'seedModelC880.py')
MODEL = "model"
MODEL_PATH = "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models/llmge_models")
TRAIN_FILE = os.path.join(SOTA_ROOT, "evalC880.py")

OUTPUT_DIR = "c880_test"
PORT = 8137

CLUSTER = "pace-ice"
LLM_MODEL = 'llama3.3'
ENVIRONMENT_DIR = os.path.join(ROOT_DIR, ".venv")
SLURM_CONFIG_DIR = os.path.join(ROOT_DIR, "slurm-config/")
LOCAL_LLM = True
HOSTNAME_DIR = os.path.join(ROOT_DIR, "hostname.log")

QC_CHECK_BOOL = False
HUGGING_FACE_BOOL = False
INFERENCE_SUBMISSION = True
CUF_TIMEOUT = 20000

LOCAL = False
if LOCAL:
    RUN_COMMAND = 'bash'
    DELAYED_CHECK = False
else:
    RUN_COMMAND = 'sbatch'
    DELAYED_CHECK = True

MACOS = platform.system() == "Darwin"
RUNLINE_AMP = ''

try:
    import torch
    if torch.mps.is_available():
        DEVICE = 'mps'
        MACOS = True
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
except ImportError:
    DEVICE = 'cpu'

# resolves to {MODEL}_{gene_id}
RUNLINE_TMP = '{}_{}'
EVAL_RUNLINE = "uv run python {} --model {} --variant_dir {VARIANT_DIR}"

"""
Evolution Constants/Params
"""
FITNESS_WEIGHTS = (1.0, -1.0)   # minimise correctness error, minimise area
INVALID_FITNESS_MAX = tuple([float(x * np.inf * -1) for x in FITNESS_WEIGHTS])
PLACEHOLDER_FITNESS = tuple([int(x * 9999999999 * -1) for x in FITNESS_WEIGHTS])
NUM_EOT_ELITES = 10
GENERATION = 0
PROB_QC = 0.0
PROB_EOT = 0.25
num_generations = 20
start_population_size = 64
population_size = 64
crossover_probability = 0.35
mutation_probability = 0.8
num_elites = 20
hof_size = 50

DNA_TXT = ""
