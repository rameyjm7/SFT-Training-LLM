#!/bin/bash

# ---------- SLURM commands ----------------------------------
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks=2	            # Request 1 task
#SBATCH --cpus-per-task=8	    # Request 8 cpu's per task
#SBATCH --time=30:00:00             # Set a 30-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=ece_6514          # Your class-specific account
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH --mem=64G            	    # memory requested
#SBATCH -J yaml_baseline_training
#SBATCH -D /home/bjanson/projects/P1/LLaMA-Factory	# the working directory
#SBATCH -o /home/bjanson/projects/P1/LLaMA-Factory/logs/%x-%j.out	# create output logs
#SBATCH -e /home/bjanson/projects/P1/LLaMA-Factory/logs/%x-%j.err	# create error logs

# Load the environment
module load Miniconda3
module load CUDA/12.6.0
source activate proj_env

# To help get performance and memory within tolerance
export TRITON_CACHE_DIR=/scratch/$USER/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# To help with 2-GPU settings
unset FORCE_TORCHRUN
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Run training, using torchrun declaration if using unset FORCE_TORCHRUN above
srun -u llamafactory-cli train sft_config.yaml
