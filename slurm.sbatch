#!/bin/bash
#SBATCH --job-name=llm_classification_full          # Job name
#SBATCH --output=output%j.txt     # Output log (%j = Job ID)
#SBATCH --error=error%j.txt       # Error log
#SBATCH --ntasks=1                 # Number of tasks (1 = serial job)
#SBATCH --cpus-per-task=32          # CPU cores per task
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00            # Time limit (hh:mm:ss)
#SBATCH --mail-type=END,FAIL       # Notifications for job done & fail
#SBATCH --mail-user=your_email@example.com  # Send-to email address

# If using conda
source ~/miniconda3/bin/activate
conda activate ftc

cd $HOME/DE_for_UQ/notebooks/Ensembling_Finetuned_LLMs
# specify the path to your script
python -u main_iclr.py --data_version extended
