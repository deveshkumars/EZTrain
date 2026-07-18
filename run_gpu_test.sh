#!/bin/bash
set -x
module load ffmpeg
cd /oscar/scratch/dkumar23/EZTrain
nvidia-smi -L
uv run main.py --train --eval
