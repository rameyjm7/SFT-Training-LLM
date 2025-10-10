#!/bin/bash

# --------  SLURM setup commands -----------
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks=1         	    # Request 1 task
#SBATCH --cpus-per-task=4     	    # Request 4 cpu's per task
#SBATCH --time=2:00:00              # Set a 2-hour time limit
#SBATCH --partition=a100_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G             # memory requested
#SBATCH -J mmlu_benchmark_finetuneAce
#SBATCH -D /home/bjanson/projects/P1/LLaMA-Factory	# the working directory
#SBATCH -o /home/bjanson/projects/P1/LLaMA-Factory/logs/%x-%j.out	# create output logs
#SBATCH -e /home/bjanson/projects/P1/LLaMA-Factory/logs/%x-%j.err	# create error logs

# Load the environment
module load Miniconda3
module load CUDA/12.6.0
source activate base_eval_env

# Define and ensure creation of directories for data input and output
COMMON_PATH="/home/bjanson/projects/P1/LLaMA-Factory"
OUTPUT_DIR="$COMMON_PATH/logs"	# Where our .json will be printed
mkdir -p "$OUTPUT_DIR"

# Set up benchmark evaluation against Qwen model
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1
MODEL="/home/bjanson/projects/P1/LLaMA-Factory/logs/qwen25_3b_instruct_h200/checkpoint-6000"
TASKS="lighteval|mmlu_redux_2|0"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Note, the execution of the lighteval vllm $MODEL_ARGS $TASKS --save-details --output-dir $OUTPUT_DIR
#  command usually here is now moved to python code below as argv = ["lighteval"....

# Convert output to .json because there is an ArrowInvalid error otherwise
export MODEL_ARGS TASKS OUTPUT_DIR
BOOT="$COMMON_PATH/le_bootstrap.py"	# Creating this to make our own .json
mkdir -p "$(dirname "$BOOT")"	# Create directory if it doesn't already exist

cat > "$BOOT" <<'PY'	# Create python code section here
import os, sys, runpy, json, subprocess

# Fix format of the data, convert to strings for JSON friendly use
def fix_data():
    try:
        from datasets import arrow_dataset as arr_data
        data_initial = arr_data.Dataset.from_list.__func__
        def repaired_data(cls, mapping, *args, **kwargs):
            try:
                return data_initial(cls, mapping, *args, **kwargs)
            except Exception:
                fixed_q_and_a = []
                for row in mapping:
                    updated_row = {}
                    for k, v in row.items():
                        #print("k: ",k)
                        #print("v: ",v)
                        if isinstance(v, set):
                            v = list(v)
                        if isinstance(v, (list, tuple, dict)): # if unsupported type found
                            #print("  v is instance to convert")
                            updated_row[k] = json.dumps(v, ensure_ascii=False) # convert to string
                        else:
                            #print("     v is NOT instance")
                            updated_row[k] = str(v) # string good as is
                    fixed_q_and_a.append(updated_row)
                return data_initial(cls, fixed_q_and_a, *args, **kwargs)
        arr_data.Dataset.from_list = classmethod(repaired_data)
    except Exception as e:
        print("Error patching:", e)

def main():
    MODEL_ARGS = os.environ.get("MODEL_ARGS")
    TASKS = os.environ.get("TASKS")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR")

    fix_data()
    try:
        help_txt = subprocess.run([sys.executable,"-m","lighteval","--help"], 
		capture_output=True, text=True).stdout
    except Exception:
        pass
    sys.argv = ["lighteval","vllm", MODEL_ARGS, TASKS, "--output-dir", OUTPUT_DIR] # Call CLI
    runpy.run_module("lighteval", run_name="__main__")

if __name__ == "__main__":
    main()
PY

# Run python and the boot program
PYTHONNOUSERSITE=1 "$(command -v python)" "$BOOT"	
echo "Program complete and files placed in -> $OUTPUT_DIR"
