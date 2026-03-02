#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "A100-80GB|H100|H200"
#SBATCH --mem 160G
#SBATCH -c 16
echo "launching LLM Server"

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"

# Check to kill any existing process on port 8137
# echo "Checking for existing server on port 8137..."
# existing_pid=$(lsof -ti:8137 2>/dev/null || true)
# if [ ! -z "$existing_pid" ]; then
#     echo "Killing existing process $existing_pid on port 8137"
#     kill -9 $existing_pid
#     sleep 2
# fi


echo "Starting LLM server on host: $SERVER_HOSTNAME"

uv run uvicorn server:app --host $SERVER_HOSTNAME --port 8137 --workers 1
