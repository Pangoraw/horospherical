import torch
import networkx as nx

from gromov_protos import is_leaf


class AverageHierarchicalCost(torch.nn.Module):
    def __init__(self, G: nx.DiGraph):
        super().__init__()
        leaves = [n for n in G.nodes() if is_leaf(G, n)]
        self.register_buffer("D", torch.zeros((len(leaves), len(leaves))))
        uG = nx.Graph(G)
        for i, la in enumerate(leaves):
            for j, lb in enumerate(leaves):
                self.D[i,j] = nx.shortest_path_length(uG, la, lb)
        self.register_buffer("S", torch.tensor(0.0, dtype=torch.float32))
        self.n = 0

    @torch.no_grad()
    def fit(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        for (z, y) in zip(pred, gt):
            self.S += self.D[z, y]
            self.n += 1

    def score(self) -> torch.Tensor:
        return self.S / self.n
