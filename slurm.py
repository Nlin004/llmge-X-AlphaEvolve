import os
import subprocess
import yaml
from src.cfg import constants

def replace_script_configuration(file_path, new_config):
        
    with open(file_path, 'w') as f:
        f.write(new_config if new_config.endswith('\n') else new_config + '\n')

def save_to_yaml(llm, python, gpu, local_llm_server, file_path=constants.SLURM_CONFIG_DIR):

        yaml_data = {"gpu_selection":gpu, "python_bash_script":python,  "llm_bash_script":llm}
        
        with open(f'{file_path}/slurm_config.yaml', 'w') as f:
                yaml.dump(yaml_data, f, indent=4)

if __name__ == "__main__":

    configuration_path = f'{constants.SLURM_CONFIG_DIR}{constants.CLUSTER}.txt'

    with open(configuration_path, 'r') as file:

        content = [item.strip() for item in file.readlines()]

        indices = [index for index, value in enumerate(content) if value == "------"]

        runsh_config_lines = "\n".join(content[indices[0]+1:indices[1]])

        run_sh = f"""
echo "launching LLM Guided Evolution"
hostname
module load uv

export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py aevolve_checkpoints
"""
        replace_script_configuration("run.sh", runsh_config_lines + run_sh)

        mixtsh_config_lines = "\n".join(content[indices[2]+1:indices[3]])
        
        mixt_sh = f"""
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
module load uv
source ~/.bashrc
# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false
uv run python llm_crossover.py '{constants.SEED_NETWORK}' '{constants.SOTA_ROOT}/models/Menghao/model_x.py' '{constants.SOTA_ROOT}/models/Menghao/model_z.py'  --top_p 0.15   --temperature 0.1 --apply_quality_control 'True' --bit 8
"""
        replace_script_configuration("src/mixt.sh", mixtsh_config_lines + mixt_sh)

        llm_gpu = content[indices[4]+1:indices[5]][0].strip(" \n")

        python_script = "\n".join(content[indices[6]+1:indices[7]]) + f"""
echo "Launching Python Evaluation"
hostname

module load cuda
module load uv
export CUDA_VISIBLE_DEVICES=0

# Run Python script
{{}}
"""
        llm_bash_config = "\n".join(content[indices[8]+1:indices[9]])
    
        llm_script =  llm_bash_config +  f"""
echo "Launching AIsurBL"
hostname

module load cuda
export CUDA_VISIBLE_DEVICES=0

# Run Python script
{{}}
"""     

        local_llm_server = f"""
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
echo "Starting LLM server on host: $SERVER_HOSTNAME"

uv run uvicorn server:app --host $SERVER_HOSTNAME --port {constants.PORT} --workers 1
"""           

        local_llm_server_config = "\n".join(content[indices[10]+1:indices[11]])

        print(local_llm_server_config)

        replace_script_configuration("server.sh", local_llm_server_config + local_llm_server)

        save_to_yaml(llm_script, python_script, llm_gpu, local_llm_server)
