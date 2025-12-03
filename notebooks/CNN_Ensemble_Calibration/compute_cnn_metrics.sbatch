#!/bin/bash
#SBATCH --job-name=cnn_metrics          # Job name
#SBATCH --output=cnn_metrics_output%j.txt     # Output log (%j = Job ID)
#SBATCH --error=cnn_metrics_error%j.txt       # Error log
#SBATCH --ntasks=1                 # Number of tasks (1 = serial job)
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --mem-per-cpu=4G           # 32GB total memory
#SBATCH --time=5:00:00             # Time limit (2 hours - metrics computation is fast)
#SBATCH --mail-type=END,FAIL       # Notifications for job done & fail
#SBATCH --mail-user=your_email@example.com  # Send-to email address

# If using conda
source ~/miniconda3/bin/activate
conda activate ftc

cd $HOME/DE_for_UQ/notebooks/CNN_Ensemble_Calibration

# Run CNN metrics computation
# Specify date to match the experimental results file
python -u compute_cnn_metrics.py --date 2025-11-30
