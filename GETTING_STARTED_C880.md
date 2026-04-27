# C880 Circuit Evolution Workflow

This guide provides a streamlined setup to run the LLM-Guided Evolution (LLM-GE) specifically for the **C880 ISCAS-85 Circuit Design** on the PACE ICE computing cluster.

Because we manage packages via `uv`, spinning this up from scratch takes just a few steps.

## 1. Prepare Your Workspace on Scratch
PACE ICE restricts quota limits in your home directory, so we will use the `scratch` partition to safely store temporary files and large models.

```bash
# Route your workspace and caches to the scratch partition
export MY_SCRATCH=/home/hice1/$(whoami)/scratch

mkdir -p $MY_SCRATCH/.cache/uv \
         $MY_SCRATCH/.cache/huggingface \
         $MY_SCRATCH/tmp \
         $MY_SCRATCH/.conda/envs

export UV_CACHE_DIR=$MY_SCRATCH/.cache/uv
export XDG_CACHE_HOME=$MY_SCRATCH/.cache
export HF_HOME=$MY_SCRATCH/.cache/huggingface
export TMPDIR=$MY_SCRATCH/tmp

# Navigate to your project
cd $MY_SCRATCH/llmge-X-AlphaEvolve
```

## 2. Install Dependencies via UV
Our `pyproject.toml` guarantees identical environments natively. We will also install `iverilog` (Icarus Verilog), which the evaluator requires, into a small isolated Conda environment.

```bash
# Install all Python dependencies through UV
module load uv
uv sync

# Install iverilog to an isolated environment
export CONDA_ENVS_PATH=$MY_SCRATCH/.conda/envs
CONDA_NO_PLUGINS=true conda create -y --solver=classic -p $MY_SCRATCH/.conda/envs/iverilog-tools -c conda-forge iverilog
```

## 3. Launch the Local LLM Server (GPU)
The AI server is required to process requests and generate Verilog circuits. Start it on a GPU node:

```bash
sbatch --job-name=c880_server -t 8:00:00 --nodes=1 -G 2 --mem=160G -c 16 \
  --wrap="cd $PWD && export CUDA_VISIBLE_DEVICES=0,1 && export SERVER_HOSTNAME=\$(hostname) && echo \$SERVER_HOSTNAME > hostname.log && uv run uvicorn server:app --host \$SERVER_HOSTNAME --port 8137 --workers 1"
```
*Tip: Use `squeue` to check the status. Once it is running, verify it's active by tracking the output via `tail -f slurm-<job_id>.out`.*

## 4. Run the Evolutionary Loop (CPU)
With the server online, trigger the actual Genetic Algorithm to begin optimizing the C880 circuit variants!

```bash
# First, include the iverilog toolchain in your path
conda activate $MY_SCRATCH/.conda/envs/iverilog-tools
export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:$PATH

# Run the evolution controller
sbatch --export=ALL --job-name=c880_run -t 8:00:00 --mem=16G -c 4 -N 1 \
  --wrap="cd $PWD && export PATH=$MY_SCRATCH/.conda/envs/iverilog-tools/bin:\$PATH && uv run run_improved.py c880_checkpoints"
```

Check points will automatically save to `c880_checkpoints/`, while resulting outputs and stats will save out to `sota/ACircuitDesign/results/`.