#!/bin/bash
#SBATCH --job-name=SPD_gen_synth          # Name of your job
#SBATCH --output=output/slurm_outputs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=output/slurm_outputs/%x_%j.err             # Error file
#SBATCH --partition=CPU              
#SBATCH --gpus-per-node=0             
#SBATCH --cpus-per-task=32             
#SBATCH --mem=256G                     
#SBATCH --time=16:00:00               

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
srun python src/data_utils/synthetic/atomistic_generator.py

echo "Job finished at: $(date)"