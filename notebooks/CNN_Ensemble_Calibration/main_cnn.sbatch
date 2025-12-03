#!/bin/bash
#SBATCH --job-name=cnn_ensemble_calib          # Job name
#SBATCH --output=cnn_output%j.txt     # Output log (%j = Job ID)
#SBATCH --error=cnn_error%j.txt       # Error log
#SBATCH --ntasks=1                 # Number of tasks (1 = serial job)
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem-per-cpu=16G           # 64GB total memory
#SBATCH --time=5:00:00             # Time limit (hh:mm:ss) - likely faster than LLM
#SBATCH --mail-type=END,FAIL       # Notifications for job done & fail
#SBATCH --mail-user=your_email@example.com  # Send-to email address

# If using conda
source ~/miniconda3/bin/activate
conda activate ftc

cd $HOME/DE_for_UQ/notebooks/CNN_Ensemble_Calibration

# Run CNN ensemble calibration experiments
# Options: --dataset MNIST, FMNIST, CIFAR, CIFAR100, MNIST10Epochs10_5L2Reg, MNIST40Epochs10_4L2Reg, or all (default: all)
# Use DATASET environment variable if set, otherwise default to 'all'
DATASET=${DATASET:-all}
python -u main_cnn.py --dataset $DATASET
