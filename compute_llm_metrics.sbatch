#!/bin/bash
#SBATCH --job-name=llm_metrics          # Job name
#SBATCH --output=comp_metrics_output%j.txt           # Output log (%j = Job ID)
#SBATCH --error=comp_metrics_error%j.txt             # Error log
#SBATCH --ntasks=1                      # Number of tasks (1 = serial job)
#SBATCH --cpus-per-task=32              # CPU cores per task
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00                 # Time limit (hh:mm:ss)
#SBATCH --mail-type=END,FAIL            # Notifications for job done & fail
#SBATCH --mail-user=your_email@example.com  # Send-to email address

# If using conda
source ~/miniconda3/bin/activate

cd $HOME/DE_for_UQ
# Run the metric computation script
python -u compute_llm_metrics.py --date 2025-09-23