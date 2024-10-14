import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import geoopt


class LinearLayer(nn.Module):
    """
    Simple linear layer wrapper with an API similar to the other classifiers in this file.
    """

    def __init__(
        self,
        n_in_feature: int,
        n_decisions: int,
    ):
        super().__init__()
        self.n_features = n_in_feature
        self.n_decisions = n_decisions

        self.linear = nn.Linear(n_in_feature, n_decisions)

        def weight_init_kaiming(m):
            class_names = m.__class__.__name__
            if class_names.find('Conv') != -1:
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif class_names.find('Linear') != -1:
                torch.nn.init.kaiming_normal_(m.weight.data)

        #init.constant_(m.bias.data, 0.0)
        self.linear.apply(weight_init_kaiming)

    def forward(self, x):  # x is a batch of size [batch_size, n_features]
        return torch.sigmoid(
            self.logits(x)
        )  # output is of size [batch_size, n_decisions]

    def logprobs(self, x):  # x is a batch of size [batch_size, n_features]
        return F.logsigmoid(
            self.logits(x)
        )  # output is of size [batch_size, n_decisions]

    def logits(self, x):
        return self.linear(x) # [batch_size, n_decisions]


class HyperbolicLayer(nn.Module):
    """
    Multi Hyperbolic Gyroplans as presented in Ganea et al., 2018
    Its inputs must already be present on the Poincaré ball.
    """

    points: geoopt.ManifoldParameter
    normals: torch.nn.Parameter

    def __init__(
        self, 
        n_in_feature: int,
        n_decisions: int,
        ball = geoopt.PoincareBall(),
    ):
        super().__init__()
        self.ball = ball
        points = self.ball.expmap0(
            0.01 * torch.randn(n_decisions, n_in_feature)
        )  # (n_decs, d)
        self.register_parameter(
            "points", geoopt.ManifoldParameter(points, manifold=self.ball)
        )

        normals = F.normalize(torch.randn(n_decisions, n_in_feature), dim=-1)
        self.register_parameter("normals", nn.Parameter(normals))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits(h).sigmoid()

    def logprobs(self, h: torch.Tensor) -> torch.Tensor:
        return F.logsigmoid(self.logits(h))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        # h is a tensor of size (b, d) on the Poincaré Ball
        # h = self.ball.expmap0(x)
        normals = self.ball.transp0(self.points, self.normals)
        mus = self.ball.dist2plane(
            h.unsqueeze(1).unsqueeze(1),  # (b, 1, 1, d)
            a=normals,
            p=self.points,
            signed=True,
            scaled=True,
        )
        return mus  # out is a tensor of size (b, n_decisions)


def busemann(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Returns the busemann distance of each sample of z with
    ideal prototype of p.
    Parameters
    ==========
        z: Tensor - size (b, d)
        p: Tensor - size (p, d)
    Returns
    =======
        C: Tensor - size (b, p)
    """
    return torch.log(torch.cdist(z, p, p=2.0).pow(2)) - torch.log(
        1 - l2(z, keepdim=True) + 1e-15
    )


def l2(z, keepdim=False):
    return z.pow(2).sum(-1, keepdim=keepdim)


class HorosphericalLayer(torch.nn.Module):
    """
        HorosphericalLayer(n_in_feature: int, n_decisions: int, phi: float = 0, proto_file: str = "")

    An horospherical classifier which can be used to classify hyperbolic samples. Input tensors *must*
    already be embeded in the Poincaré ball.
    """

    point: geoopt.ManifoldParameter
    bias: torch.nn.Parameter

    def __init__(
        self,
        n_in_feature: int,
        n_decisions: int,
        phi: float = 0.,
        proto_file:str="",
    ):
        super().__init__()

        print(f">> Using precomputed prototypes as initialization from '{proto_file}'")
        point = torch.from_numpy(np.load(proto_file)).float()
        assert point.shape == (n_decisions, n_in_feature), (point.shape, (n_decisions, n_in_feature))
        point = point.reshape(n_decisions, n_in_feature)

        sphere = geoopt.Sphere()
        self.register_parameter("point", geoopt.ManifoldParameter(point, sphere))

        bias = torch.zeros(n_decisions)
        self.register_parameter("bias", torch.nn.Parameter(bias))

        self.phi = phi
        self.penalize = self.phi != 0.

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

    def logprobs(self, x):
        return torch.nn.functional.logsigmoid(self.logits(x))

    def logits(self, x):
        D = busemann(x, self.point)  # size (b, n_decisions)
        if self.penalize:
            penalty = (
                self.phi *
                torch.log(1 - l2(x, keepdim=True) + 1e-15)
            )  # size (b, 1)
            return ((-D + self.bias) + penalty)
        return (-D + self.bias)  # size (b, n_decisions)

    def radius(self) -> torch.Tensor:
        """
        Returns the radii of the Horospheres by solving the equation -D+a = 0.
        along the Poincare ball radius without penalization.
        One of the roots of the equation is always 1.0 corresponding to `self.point`,
        we are interested in the other one.
        """
        x0 = -torch.tanh(self.bias / 2)
        diameter = (1 - x0).abs()
        radius = diameter / 2
        return radius


class BusemannPrototypes(nn.Module):
    """
    Busemann classifier implementation based on Ghadimi Atigh et al.

    > "Hyperbolic Busemann Learning with Ideal Prototypes"
    > Mina Ghadimi Atigh, Martin Keller-Ressel, Pascal Mettes
    > Neurips, 2021
    """

    prototypes: torch.Tensor

    def __init__(self, n_in_feature: int, n_decisions: int, proto_file:str=""):
        super().__init__()
        # point_path = f"prototypes/prototypes{proto_type}-{n_in_feature}d-{n_decisions}c.npy"
        point_path = proto_file
        print(f">> Using precomputed prototypes as initialization from '{point_path}'")
        self.register_buffer(
            "prototypes",
            torch.from_numpy(np.load(point_path)).float()
        )
        assert self.prototypes.shape == (n_decisions, n_in_feature), "invalid prototype file"

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

    def logprobs(self, x):
        return torch.nn.functional.logsigmoid(self.logits(x))

    def logits(self, x, y=None):
        if y is None:
            return -busemann(x, self.prototypes)

        y_protos = self.prototypes[y, :]
        assert y_protos.shape == x.shape
        return (
            -torch.log((y_protos - x).pow(2).sum(-1)) +
            torch.log(1 - x.pow(2).sum(-1))
        )
