#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -A tueng010b
#SBATCH -p GpuQ

python3 main_fixed_intents_labeled_ratio.py