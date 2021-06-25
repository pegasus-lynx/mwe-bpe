#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job_name=rtg
#SBATCH --error=rtg.err
#SBATCH --output=rtg.out
#SBATCH --partition=gpu:1

module load cuda/10.1
module load conda
module load anaconda/3/python3.7
source activate bpe
cd parzival/runs
rtg-pipe -w 8k_ngdf_r100
source deactivate