#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:3
#SBATCH -C "A100-40GB|A100-80GB|H100|H200|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem 10G
#SBATCH -c 48
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
module load uv
source ~/.bashrc
# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false
uv run python llm_crossover.py '/storage/ice1/6/9/nlin47/AlphaEvolve/llm-guided-evolution-fork/sota/AEvolve/seedModel1.py' '/storage/ice1/6/9/nlin47/AlphaEvolve/llm-guided-evolution-fork/sota/AEvolve/models/Menghao/model_x.py' '/storage/ice1/6/9/nlin47/AlphaEvolve/llm-guided-evolution-fork/sota/AEvolve/models/Menghao/model_z.py'  --top_p 0.15   --temperature 0.1 --apply_quality_control 'True' --bit 8
