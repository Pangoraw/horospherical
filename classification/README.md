# Reproducing experiments

The first step is to instantiate the provided environment at the top-level.

Secondly, one needs to have access to the relevant uniform prototypes (optionally, they can be positioned in a hierarchically-informed manner).

Example with CIFAR10 2d with a horospherical classifier with uniform initial prototypes:

```bash
python classification/train.py \
    --dataset cifar10 \
    --method horospherical \
    --proto_file prototypes/prototypesuniform-2d-10c.npy \
    --dim 2 \
    --epochs 1110 \
    --schedule none \
    --lambda_ 0.75 \
    --weight_decay 0.00005 \
    --lr 0.0005
```
