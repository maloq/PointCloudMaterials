#!/bin/bash
#SBATCH --job-name=SPD_72        # Name of your job
#SBATCH --output=output/slurm_outputs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=output/slurm_outputs/%x_%j.err             # Error file
#SBATCH --partition=H100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gpus-per-node=2             # Request 1 node
#SBATCH --gres=gpu:2                  # Request 1 GPU
#SBATCH --cpus-per-task=16             # Request 8 CPU cores
#SBATCH --mem=94G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

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
# srun python src/training_methods/spd/train_spd.py --config-name spd
srun python src/training_methods/spd/train_spd.py --config-name spd_synth latent_size=72
echo "Job finished at: $(date)"