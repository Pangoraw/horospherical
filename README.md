# Horospherical Learning with Smart Prototypes

> [!IMPORTANT]
> 🚧 This repository is still in construction.

This repository contains the code for the article "Horospherical Learning with Smart Prototypes" to be published at [BMVC2024](https://bmvc2024.com).

## Generating uniform prototypes

In order to generate uniformly distributed prototypes on the hypersphere (_i.e._ the boundary of the Poincaré ball), we use the same technique as the one used by Ghadimi et al. and Wang et Isola. which is to optimize a set of points on the hypersphere by maximizing the pairwise distances between points.

```bash
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

## Making smart prototypes

In order to assign these randomly generated prototypes according to the method presented in the article, one need to have a label hierarchy and a set of prototypes (one for each node in the hierarchy).

```bash
DATASET=cub
python gromov_protos.py \
    -d $D \
    --dataset $DATASET \
    --input-protos prototypesuniform_${D}d_${C}c.npy \
    --output-file prototypesgromov_${D}d_200c.npy
```

## Dependencies

This code depends on the following python packages.

 - [Torch](https://pytorch.org)
 - [PythonOT](https://github.com/PythonOT/POT)
 - NetworkX
 - tqdm
 - pandas
 - numpy
 - matplotlib
