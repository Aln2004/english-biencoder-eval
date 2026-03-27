#!/bin/bash

#SBATCH --job-name=multilleqa                    # Job name
#SBATCH --output=multilleqa.txt                  # Output file
#SBATCH --ntasks=1                               # Run a single task
#SBATCH --time=72:00:00                          # Time limit hh:mm:ss
#SBATCH --partition=gpu-a100                     # Specify the gpu-a100 partition
#SBATCH --gres=gpu:2                             # Request 2 GPU
#SBATCH --nodelist=node1                         # Ensure it uses node2, where gpu-h100 is available

python3 -u multilingual_lleqa.py
