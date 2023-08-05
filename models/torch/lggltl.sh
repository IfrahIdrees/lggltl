#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -o train.out
#SBATCH -e train_error.out
#SBATCH -t 15:05:00

# Load CUDA module
# module load cuda
# module load cuda/11.3.1
# module load cudnn
module load python/3.7.4
# conda init bash
# module load anaconda
# conda init bash
# module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
# conda env list
# echo ". /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh" >> ~/.bashrc
# conda activate /users/iidrees/anaconda3/envs/lggltl-start2
conda activate lggltl-start2
# conda activate /users/iidrees/anaconda3/envs/copynet

# Compile CUDA program and run
# conda activate lggltl
# python torch_seq2seq.py 200
python torch_seq2seq.py lang2ltl --src_dir_path ../../data/osm/lang2ltl/boston/ --is_load True