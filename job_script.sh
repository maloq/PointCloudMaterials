#!/bin/bash
#SBATCH --job-name=PnAE_l8            # Name of your job
#SBATCH --output=output/slurm_outputs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=output/slurm_outputs/%x_%j.err             # Error file
#SBATCH --partition=A100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gpus-per-node=1             # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=32G                     # Request 32 GB of memory
#SBATCH --time=8:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet

# Change to the project directory and set PYTHONPATH
cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=$PYTHONPATH:/home/infres/vmorozov/PointCloudMaterials

# Run the Python script
srun python src/autoencoder/train_autoencoder.py --config-name autoencoder_PnAEFold  

echo "Job finished at: $(date)"