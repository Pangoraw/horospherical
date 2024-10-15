# Horospherical Learning with Smart Prototypes

> [!IMPORTANT]
> 🚧 This repository is still in construction.

<img width="1331" alt="overview" src="https://github.com/user-attachments/assets/4bab9ca6-68b7-48b2-a309-0fd45f2d86f0">

This repository contains the code for the article "Horospherical Learning with Smart Prototypes" to be published at [BMVC2024](https://bmvc2024.org).

## Generating uniform prototypes

In order to generate uniformly distributed prototypes on the hypersphere (_i.e._ the boundary of the Poincaré ball), we use the same technique as the one used by [Ghadimi Atigh et al.](https://proceedings.neurips.cc/paper/2021/hash/01259a0cb2431834302abe2df60a1327-Abstract.html) and [Wang et Isola.](http://proceedings.mlr.press/v119/wang20k/wang20k.pdf) which is to optimize a set of points on the hypersphere by maximizing the pairwise distances between points.

```sh
C=252 # Number of classes
D=256 # Number of dimensions
EPOCHS=100_000 # Number of epochs

python prototype_learning.py \
    -c $C \
    -d $D \
    -e $EPOCHS \
    -r prototypesuniform_${D}d_${C}c.npy
```

After some number crunching, it will generate the file `prototypesuniform_256d_10c.npy` on disk.

## Making "smart" prototypes

In order to assign these randomly generated prototypes according to the method presented in the article, one need to have a label hierarchy and a set of prototypes (one for each node in the hierarchy).

```sh
DATASET=cub
python gromov_protos.py \
    -d $D \
    --dataset $DATASET \
    --input-protos prototypesuniform_${D}d_${C}c.npy \
    --output-file prototypesgromov_${D}d_200c.npy
```
## Horospherical Learning

The code for the horospherical classifier presented in the article is located in the file [`horospherical.py`](./horospherical.py).

## Experiments

In the paper, several kind of experiments on classification over hierarchical data is performed.

### Image Classification

The code for the image classification experiments is located in folder [`classification/`](./classification).

The following datasets have been used:

 - [CUB200](https://data.caltech.edu/records/65de6-vp158) - should be downloaded and extracted in the `data/` folder under the name `data/CUB_200_2011/`.
 - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) - will be automatically downloaded in the `data` folder.
 - [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) - will be automatically downloaded in the `data` folder.

### Semantic Segmentation

TO BE RELEASED.

### Point-Cloud Segmentation

TO BE RELEASED.

## Dependencies

This code depends on the following python packages.

 - [pytorch](https://pytorch.org)
 - [PythonOT](https://github.com/PythonOT/POT)
 - [geoopt](https://github.com/geoopt/geoopt)
 - NetworkX
 - tqdm
 - pandas
 - numpy
 - matplotlib

A conda environment is available in the [`env.yml`](./env.yml) file to reproduce the same versions.

## Citing

If this repository or the article is helpful to your research, considering citing the corresponding article:

```bibtex
@inproceedings{berg2024horospherical,
  title={Horospherical Learning with Smart Prototypes},
  author={Berg, Paul and Michele, Bjoern and Pham, Minh-Tan and Chapel, Laetitia and Courty, Nicolas},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2024}
}
```
