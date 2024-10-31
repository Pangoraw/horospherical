## Reproducing

```
DIM=50

# CUB200

python train.py \
    --dataset cub \
    --method horospherical \
    --proto_file "<PATH_TO_PROTOTYPES>" \
    --dim $DIM \
    --epochs 1000 \
    --schedule none \
    --lambda_ 0.0 \
    --optimizer sgd \
    --lr 0.045 \
    --momentum 0.2 \
    --weight_decay 0.0

# CIFAR10

python train.py \
    --dataset cifar10 \
    --method horospherical \
    --proto_file "<PATH_TO_PROTOTYPES>" \
    --dim $DIM \
    --epochs 1110 \
    --schedule none \
    --lambda_ 0.75 \
    --weight_decay 0.00005 \
    --lr 0.0005

# CIFAR100

python train.py \
    --dataset cifar100 \
    --method horospherical \
    --proto_file "<PATH_TO_PROTOTYPES>" \
    --dim $DIM \
    --epochs 1110 \
    --schedule none \
    --lambda_ 0.1 \
    --weight_decay 0.00005 \
    --lr 0.0005
```
