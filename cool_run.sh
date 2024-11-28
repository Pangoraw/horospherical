#!/bin/bash

#SBATCH --time=2-0
#SBATCH --gres=gpu:1
#SBATCH -c 9
#SBATCH --mem-per-cpu=3G
#SBATCH --partition shortrun

set -o errexit
set -o pipefail
set -o nounset

# ---

DIM=$1

/share/castor/home/berg/micromamba/envs/pt/bin/python \
    classification/train.py \
        --dataset cub \
        --method horospherical \
        --proto_file numbered_prototypes/prototypesgromov0-${DIM}d-200c.npy \
        --dim ${DIM} \
        --epochs 2110 \
        --schedule none \
        --lambda_ 0.05 \
        --weight_decay 0.0001 \
        --lr 0.0005 \
        --online_loss 1.0
