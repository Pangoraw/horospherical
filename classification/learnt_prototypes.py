"""
Vendored code from:
    https://github.com/VSainteuf/metric-guided-prototypes-pytorch/blob/master/torch_prototypes/modules/prototypical_network.py

from "Leveraging Class Hierarchies with Metric-Guided Prototype Learning" Garnot et Landrieu. 2021.
"""
import torch
import torch.nn as nn
import numpy as np

class Eucl_Mat(nn.Module):
    """Pairwise Euclidean distance"""

    def __init_(self):
        super(Eucl_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Euclidean distances

        """
        return torch.norm(mapping[:, None, :] - mapping[None, :, :], dim=-1)


class Cosine_Mat(nn.Module):
    """Pairwise Cosine distance"""

    def __init__(self):
        super(Cosine_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Cosine distances

        """
        return 1 - nn.CosineSimilarity(dim=-1)(mapping[:, None, :], mapping[None, :, :])


class Pseudo_Huber(nn.Module):
    """Pseudo-Huber function"""

    def __init__(self, delta=1):
        super(Pseudo_Huber, self).__init__()
        self.delta = delta

    def forward(self, input):
        out = (input / self.delta) ** 2
        out = torch.sqrt(out + 1)
        out = self.delta * (out - 1)
        return out



class LearntPrototypes(nn.Module):
    """
    Learnt Prototypes Module. Classification module based on learnt prototypes, to be wrapped around a backbone
    embedding network.
    """

    def __init__(
        self,
        n_prototypes,
        embedding_dim,
        prototypes=None,
        squarred=False,
        ph=None,
        dist="euclidean",
        device="cuda",
    ):
        """

        EDIT: removed model parameter/attribute.

        Args:
            n_prototypes (int): number of prototypes to use
            embedding_dim (int): dimension of the embedding space prototypes (tensor): Prototype tensor of shape (n_prototypes x embedding_dim), squared (bool): Whether to use the squared Euclidean distance or not ph (float): if specified, the distances function is huberized with delta parameter equal to the specified value
            dist (str): default 'euclidean', other possibility 'cosine'
            device (str): device on which to declare the prototypes (cpu/cuda)
        """
        super(LearntPrototypes, self).__init__()
        self.prototypes = (
            nn.Parameter(
                torch.rand((n_prototypes, embedding_dim), device=device)
            ).requires_grad_(True)
            if prototypes is None
            else nn.Parameter(prototypes).requires_grad_(False)
        )
        self.n_prototypes = n_prototypes
        self.squarred = squarred
        self.dist = dist
        self.ph = None if ph is None else Pseudo_Huber(delta=ph)

    def logits(self, *args, **kwargs):
        return self(*args, **kwargs)

    def forward(self, embeddings, **kwargs):
        if len(embeddings.shape) == 4:  # Flatten 2D data
            two_dim_data = True
            b, c, h, w = embeddings.shape
            embeddings = (
                embeddings.view(b, c, h * w)
                .transpose(1, 2)
                .contiguous()
                .view(b * h * w, c)
            )
        else:
            two_dim_data = False

        if self.dist == "cosine":
            dists = 1 - nn.CosineSimilarity(dim=-1)(
                embeddings[:, None, :], self.prototypes[None, :, :]
            )
        else:
            dists = torch.norm(
                embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1
            )
            if self.ph is not None:
                dists = self.ph(dists)
        if self.squarred:
            dists = dists ** 2

        if two_dim_data:  # Un-flatten 2D data
            dists = (
                dists.view(b, h * w, self.n_prototypes)
                .transpose(1, 2)
                .contiguous()
                .view(b, self.n_prototypes, h, w)
            )
        else:
            dists = dists.view(dists.size(0), 1, dists.size(-1))

        return -dists


class LossWithDistortion(nn.Module):
    def __init__(self, loss, protos, dist_loss, lambda_=1.0):
        super().__init__()
        self.loss = loss
        self.dist_loss = dist_loss
        self.protos = protos
        self.lambda_ = lambda_
    def forward(self, x, y):
        return self.loss(x,y) + self.lambda_ * self.dist_loss(self.protos)


class DistortionLoss(nn.Module):
    """Scale-free squared distortion regularizer"""

    def __init__(self, D, dist="euclidian", scale_free=True):
        super(DistortionLoss, self).__init__()
        self.D = D
        self.scale_free = scale_free
        if dist == "euclidian":
            self.dist = Eucl_Mat()
        elif dist == "cosine":
            self.dist = Cosine_Mat()

    def forward(self, mapping, idxs=None):
        d = self.dist(mapping)

        if self.scale_free:
            a = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device))
            scaling = a.sum() / torch.pow(a, 2).sum()
        else:
            scaling = 1.0

        d = (scaling * d - self.D) ** 2 / (
            self.D + torch.eye(self.D.shape[0], device=self.D.device)
        ) ** 2
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d
