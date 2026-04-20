# PACE ICE Setup for the C880 Circuit Design Workflow

This guide documents the exact PACE ICE setup for running the `ACircuitDesign`
workflow in this repository, using the same configuration that was validated in
this project.

It is written for a new user starting from scratch on Georgia Tech PACE ICE.

## What This Run Does

This workflow runs LLM-guided evolution for the ISCAS-85 `c880` circuit design
task. The active configuration is the C880 setup:

- `src/cfg/constants.py` imports `constants_C880`
- `src/cfg/constants_C880.py` points to:
  - `sota/ACircuitDesign`
  - `seedModelC880.py`
  - `evalC880.py`
  - `templates_CD`
- the evaluator writes per-gene fitness as:
  - correctness error
  - area
  - delay

## Important Assumptions

This README assumes the repo already contains the working Circuit Design code
changes. In particular:

- `src/cfg/constants.py` imports `constants_C880`
- `src/cfg/constants_C880.py` uses:
  - `SOTA_ROOT = sota/ACircuitDesign`
  - `SEED_NETWORK = seedModelC880.py`
  - `TRAIN_FILE = evalC880.py`
  - `TEMPLATE_DIR = templates_CD`
  - `OUTPUT_DIR = "c880_test"`
  - `PORT = 8137`
- `sota/ACircuitDesign/evalC880.py` writes results into
  `sota/ACircuitDesign/results`

## One Required Slurm Fix

PACE ICE rejected the original LLM mutation jobs because the repo's
`slurm-config/slurm_config.yaml` used GPU constraints that produced
`PD (BadConstraints)`.

Before running the workflow on PACE ICE, update
`slurm-config/slurm_config.yaml` so the `llm_bash_script` does not use
`#SBATCH -C {}`.

Use this shape:

```yaml
gpu_selection: ""
llm_bash_script: '#!/bin/bash

    #SBATCH --job-name=llm_oper
    #SBATCH -t 8:00:00
    #SBATCH --mem-per-gpu 16G
    #SBATCH -G 1
    #SBATCH -c 2
    #SBATCH -N 1
    echo "Launching AIsurBL"
    hostname

    module load cuda
    export CUDA_VISIBLE_DEVICES=0

    {}
    '
python_bash_script: '#!/bin/bash

    #SBATCH --job-name=evaluateGene
    #SBATCH -t 4:00:00
    #SBATCH --gres=gpu:1
    #SBATCH -G 1
    #SBATCH --mem 16G
    #SBATCH -c 12
    #SBATCH -N 1
    echo "Launching Python Evaluation"
    hostname

    module load cuda
    module load uv
    export CUDA_VISIBLE_DEVICES=0

    {}
    '
```

Without this fix, many `llm_oper` jobs can sit in the queue forever with
`BadConstraints`.

## 1. Put the Repo on Scratch

Use scratch space, not your home directory, for the working repo and caches.

If cloning from git:

```bash
export MY_SCRATCH=/home/hice1/<gtusername>/scratch
cd $MY_SCRATCH

git clone <repo-url> llmge-X-AlphaEvolve
cd llmge-X-AlphaEvolve
```

If copying from an existing repo on PACE:

```bash
export MY_SCRATCH=/home/hice1/<gtusername>/scratch
rsync -av /path/to/source/llmge-X-AlphaEvolve/ $MY_SCRATCH/llmge-X-AlphaEvolve/
cd $MY_SCRATCH/llmge-X-AlphaEvolve
```

## 2. Move Caches and Temp Files to Scratch

This step avoids quota failures in `~/.cache`, `~/.conda`, and temporary
directories.

```bash
export MY_SCRATCH=/home/hice1/<gtusername>/scratch

mkdir -p $MY_SCRATCH/.cache/uv
mkdir -p $MY_SCRATCH/.cache/huggingface
mkdir -p $MY_SCRATCH/tmp
mkdir -p $MY_SCRATCH/.conda/pkgs
mkdir -p $MY_SCRATCH/.conda/envs

export UV_CACHE_DIR=$MY_SCRATCH/.cache/uv
export XDG_CACHE_HOME=$MY_SCRATCH/.cache
export HF_HOME=$MY_SCRATCH/.cache/huggingface
export TMPDIR=$MY_SCRATCH/tmp
export CONDA_PKGS_DIRS=$MY_SCRATCH/.conda/pkgs
export CONDA_ENVS_PATH=$MY_SCRATCH/.conda/envs
```

## 3. Create the Python Environment

Load `uv`, create the virtual environment, and sync dependencies:

```bash
module load uv
uv venv .venv
source .venv/bin/activate
uv sync
```

If `uv sync` pulls too much or you only want the Circuit Design path, use a
minimal install instead:

```bash
module load uv
uv venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install \
  deap==1.4.1 \
  "numpy>=2.0.2" \
  pyyaml \
  requests \
  huggingface_hub \
  "google-genai>=1.12.1" \
  torch==2.6.0 \
  torchvision \
  transformers \
  accelerate \
  fastapi \
  uvicorn \
  pydantic \
  tqdm
```

## 4. Install `iverilog` and `vvp` in Scratch

PACE ICE did not expose `iverilog` as a module in this setup, so install it
through a scratch-local conda environment:

```bash
CONDA_NO_PLUGINS=true conda create -y --solver=classic \
  -p $MY_SCRATCH/.conda/envs/iverilog-tools \
  -c conda-forge \
  iverilog
```

Activate it whenever you run the workflow:

```bash
source .venv/bin/activate
conda activate $MY_SCRATCH/.conda/envs/iverilog-tools
export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:$PATH
module load uv
```

Quick toolchain check:

```bash
which iverilog
which vvp
```

## 5. Smoke Test the C880 Evaluator

Run the evaluator once on the seed model:

```bash
python sota/ACircuitDesign/evalC880.py \
  --model seedModelC880 \
  --variant_dir sota/ACircuitDesign \
  --save_dir c880_smoke
```

Expected success output is similar to:

```text
[SUCCESS] Correct logic! Area: 557, Delay: 385
Metrics saved: 0.0, 557, 385
Job Done
```

If this works, then:

- Python can import the seed generator
- `iverilog` is available
- `vvp` is available
- the evaluator path is correct

## 6. Start the LLM Server

Submit the GPU server job:

```bash
SERVER_JOB=$(sbatch --parsable \
  --job-name=c880_server \
  -t 8:00:00 \
  --nodes=1 \
  -G 2 \
  --mem=160G \
  -c 16 \
  --wrap="cd $PWD && export CUDA_VISIBLE_DEVICES=0,1 && export SERVER_HOSTNAME=\$(hostname) && echo \$SERVER_HOSTNAME > hostname.log && $PWD/.venv/bin/python -m uvicorn server:app --host \$SERVER_HOSTNAME --port 8137 --workers 1")

echo $SERVER_JOB
squeue -j $SERVER_JOB
```

Wait until the job state is `R`. Do not try to `tail` the output while the job
is still pending, because the log file may not exist yet.

Once it is running:

```bash
tail -f slurm-${SERVER_JOB}.out
```

Expected output includes something like:

```text
Uvicorn running on http://<hostname>.pace.gatech.edu:8137
```

## 7. Start the Main Evolution Run

Open a second shell, reactivate the same environments, and submit the CPU-side
controller job:

```bash
source .venv/bin/activate
conda activate $MY_SCRATCH/.conda/envs/iverilog-tools
export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:$PATH

RUN_JOB=$(sbatch --parsable \
  --export=ALL \
  --job-name=c880_run \
  -t 8:00:00 \
  --mem=16G \
  -c 4 \
  -N 1 \
  --wrap="cd $PWD && export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:\$PATH && $PWD/.venv/bin/python run_improved.py c880_checkpoints")

echo $RUN_JOB
squeue -j $RUN_JOB
```

## 8. Monitor the Run

Live run log:

```bash
tail -f slurm-${RUN_JOB}.out
```

Newest result files:

```bash
ls -lt sota/ACircuitDesign/results | head
```

Newest generated candidate programs:

```bash
ls -lt sota/ACircuitDesign/models/llmge_models | head
```

Newest checkpoints:

```bash
ls -lt c880_checkpoints | head
```

Newest generation artifacts:

```bash
ls -lt c880_test | head
```

## 9. Where the Fitness Is Written

Each evaluated gene writes a file into:

```text
sota/ACircuitDesign/results/
```

Each file is named:

```text
<gene_id>_results.txt
```

and contains:

```text
correctness_error, area, delay
```

Interpretation:

- `0.0` correctness error means the candidate matched the reference
- lower area is better
- lower delay is better

## 10. Where the Generated Candidates Live

Generated Python candidates are written to:

```text
sota/ACircuitDesign/models/llmge_models/
```

They are named like:

```text
model_<gene_id>.py
```

## 11. Where the Prompt and Generation Artifacts Live

Generation-specific artifacts go under:

```text
c880_test/
```

Each generation gets a numbered subdirectory that can contain:

- mutation prompt text files
- submission scripts
- per-gene intermediate files

## 12. Where the Checkpoints Live

Checkpoints are stored in:

```text
c880_checkpoints/
```

Each generation is saved as:

```text
checkpoint_gen_<N>.pkl
```

## 13. Inspect the Latest Checkpoint

To inspect the latest checkpoint's population and stored fitness:

```bash
python - <<'PY'
import glob, pickle
files = sorted(glob.glob("c880_checkpoints/checkpoint_gen_*.pkl"))
latest = files[-1]
print("Latest checkpoint:", latest)
ckpt = pickle.load(open(latest, "rb"))
pop = ckpt["population"]
g = ckpt["GLOBAL_DATA"]
print("Population size:", len(pop))
for ind in pop[:10]:
    gene = ind[0]
    print(gene, g.get(gene, {}).get("fitness"), g.get(gene, {}).get("status"))
PY
```

To inspect the best stored individuals:

```bash
python - <<'PY'
import glob, pickle
files = sorted(glob.glob("c880_checkpoints/checkpoint_gen_*.pkl"))
latest = files[-1]
ckpt = pickle.load(open(latest, "rb"))
hof = ckpt["hof"]
print("Latest checkpoint:", latest)
for ind in list(hof)[:10]:
    print(ind[0], ind.fitness.values)
PY
```

## 14. Known Failure Modes

### Disk quota exceeded

Cause:

- caches or temp files are landing in your home directory

Fix:

- re-export the scratch cache variables from Section 2

### `iverilog` not found

Cause:

- the conda environment is not activated
- `PATH` does not include the scratch-local `iverilog-tools` env

Fix:

```bash
conda activate $MY_SCRATCH/.conda/envs/iverilog-tools
export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:$PATH
```

### `tail: cannot open slurm-<jobid>.out`

Cause:

- the job is still pending and the log file has not been created yet

Fix:

- wait until `squeue` shows the job as `R`

### `PD (BadConstraints)` for `llm_oper`

Cause:

- `slurm-config/slurm_config.yaml` still includes the GPU constraint line

Fix:

- remove `#SBATCH -C {}` from the LLM submission script
- set `gpu_selection: ""`

## 15. Sleep and Disconnect Behavior

If your laptop sleeps or your SSH session disconnects:

- PACE Slurm jobs keep running
- your terminal connection is what stops

When you reconnect:

```bash
cd /home/hice1/<gtusername>/scratch/llmge-X-AlphaEvolve
squeue -u <gtusername>
```

Then tail the right logs again:

```bash
tail -f slurm-<server_job_id>.out
tail -f slurm-<run_job_id>.out
```

## 16. Optional: Separate Experimental Runs

If you want to run a second experiment simultaneously, do not use the same
working tree. Make a separate repo copy on scratch and give it its own:

- `hostname.log`
- output directory
- checkpoint directory
- server port

That avoids collisions between concurrent runs.

## Quick Start Summary

If the repo is already configured, the shortest working path is:

```bash
export MY_SCRATCH=/home/hice1/<gtusername>/scratch

mkdir -p $MY_SCRATCH/.cache/uv
mkdir -p $MY_SCRATCH/.cache/huggingface
mkdir -p $MY_SCRATCH/tmp
mkdir -p $MY_SCRATCH/.conda/pkgs
mkdir -p $MY_SCRATCH/.conda/envs

export UV_CACHE_DIR=$MY_SCRATCH/.cache/uv
export XDG_CACHE_HOME=$MY_SCRATCH/.cache
export HF_HOME=$MY_SCRATCH/.cache/huggingface
export TMPDIR=$MY_SCRATCH/tmp
export CONDA_PKGS_DIRS=$MY_SCRATCH/.conda/pkgs
export CONDA_ENVS_PATH=$MY_SCRATCH/.conda/envs

cd $MY_SCRATCH/llmge-X-AlphaEvolve

module load uv
source .venv/bin/activate
conda activate $MY_SCRATCH/.conda/envs/iverilog-tools
export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:$PATH

python sota/ACircuitDesign/evalC880.py \
  --model seedModelC880 \
  --variant_dir sota/ACircuitDesign \
  --save_dir c880_smoke

SERVER_JOB=$(sbatch --parsable \
  --job-name=c880_server \
  -t 8:00:00 \
  --nodes=1 \
  -G 2 \
  --mem=160G \
  -c 16 \
  --wrap="cd $PWD && export CUDA_VISIBLE_DEVICES=0,1 && export SERVER_HOSTNAME=\$(hostname) && echo \$SERVER_HOSTNAME > hostname.log && $PWD/.venv/bin/python -m uvicorn server:app --host \$SERVER_HOSTNAME --port 8137 --workers 1")

echo $SERVER_JOB
squeue -j $SERVER_JOB
```

Then, in a second shell:

```bash
cd $MY_SCRATCH/llmge-X-AlphaEvolve
source .venv/bin/activate
conda activate $MY_SCRATCH/.conda/envs/iverilog-tools
export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:$PATH

RUN_JOB=$(sbatch --parsable \
  --export=ALL \
  --job-name=c880_run \
  -t 8:00:00 \
  --mem=16G \
  -c 4 \
  -N 1 \
  --wrap="cd $PWD && export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:\$PATH && $PWD/.venv/bin/python run_improved.py c880_checkpoints")

echo $RUN_JOB
squeue -j $RUN_JOB
```

That is the working PACE ICE path for this repository's C880 setup.
