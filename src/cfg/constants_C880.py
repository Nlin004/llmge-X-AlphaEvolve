import os
import sys
import numpy as np
import platform
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOTA_ROOT = os.path.join(ROOT_DIR, 'sota/ACircuitDesign')
SEED_NETWORK = os.path.join(SOTA_ROOT, 'seedModelC880.py')
MODEL = "model"
MODEL_PATH = "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/"
VARIANT_DIR = os.path.join(SOTA_ROOT, "models/llmge_models")
TRAIN_FILE = os.path.join(SOTA_ROOT, "evalC880.py")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates_CD")

OUTPUT_DIR = "c880_test"
PORT = 8137
PYTHON_CMD = [sys.executable, "-u"]

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

LOCAL = shutil.which("sbatch") is None
if LOCAL:
    RUN_COMMAND = None
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
FITNESS_WEIGHTS = (-1.0, -1.0, -1.0)   # minimise correctness error, minimise area, minimise delay
INVALID_FITNESS_MAX = tuple([float(x * np.inf * -1) for x in FITNESS_WEIGHTS])
PLACEHOLDER_FITNESS = tuple([int(x * 9999999999 * -1) for x in FITNESS_WEIGHTS])
NUM_EOT_ELITES = 10
GENERATION = 0
PROB_QC = 1.0
PROB_EOT = 0.25
num_generations = 20
start_population_size = 32
population_size = 32
crossover_probability = 0.2
mutation_probability = 0.5
num_elites = 20
hof_size = 50

DNA_TXT = ""
