#!/bin/bash
#SBATCH --job-name=calc_uncertainty_iclr          # Job name
#SBATCH --output=uncertainty_output_%j.txt        # Output log (%j = Job ID)
#SBATCH --error=uncertainty_error_%j.txt          # Error log
#SBATCH --ntasks=1                                # Number of tasks (1 = serial job)
#SBATCH --cpus-per-task=16                        # CPU cores per task (less than main job)
#SBATCH --mem-per-cpu=8G                          # 8GB per CPU = 128GB total
#SBATCH --time=05:00:00                            # Time limit (5 hours should be plenty)
#SBATCH --mail-type=END,FAIL                      # Notifications for job done & fail
#SBATCH --mail-user=your_email@example.com  # Send-to email address

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate ftc

# Navigate to the script directory
cd $HOME/DE_for_UQ/notebooks/Ensembling_Finetuned_LLMs

# Run the uncertainty calculation script
python -u calc_uncertainty_iclr.py