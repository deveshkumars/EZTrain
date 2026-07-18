#!/bin/bash
#SBATCH -J eztrain-full
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /oscar/scratch/dkumar23/EZTrain/slurm-%j.out
set -x
module load ffmpeg
cd /oscar/scratch/dkumar23/EZTrain
nvidia-smi -L
uv run main.py --train --eval --timesteps 10000000 --num_evals 10 --steps 400 --video gifs/hover_10M.mp4
