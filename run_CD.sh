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
