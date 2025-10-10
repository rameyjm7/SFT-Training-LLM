#!/bin/bash

# --------  SLURM setup commands -----------
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks=1         	    # Request 1 task
#SBATCH --cpus-per-task=4     	    # Request 4 cpu's per task
#SBATCH --time=5:00:00              # Set a 2-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G             # memory requested
#SBATCH -J math_reasoning_finetuneAce
#SBATCH -D /home/bjanson/projects/P1/LLaMA-Factory	# the working directory
#SBATCH -o /home/bjanson/projects/P1/LLaMA-Factory/logs/%x-%j.out	# create output logs
#SBATCH -e /home/bjanson/projects/P1/LLaMA-Factory/logs/%x-%j.err	# create error logs

# Load the environment
module load Miniconda3
module load CUDA/12.6.0
source activate base_eval_env

# Define and ensure creation of directories for data input and output
OUTPUT_DIR="/home/bjanson/projects/P1/LLaMA-Factory/logs"
mkdir -p "$OUTPUT_DIR"

# Set up math reasoning evaluation against Qwen model
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1
MODEL="/home/bjanson/projects/P1/LLaMA-Factory/logs/qwen25_3b_instruct_h200/checkpoint-6000"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Execute the evaluation
lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0,extended|lcb:codegeneration|0|0" --save-details --output-dir $OUTPUT_DIR
