
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, get_laplacian


class Spectral(BaseTransform):
    r""" Add edge_index_ext and edge_attr_ext as object attribute and fill it with edge_index
    and edge_attr.
    """

    def __init__(self, n_max):
        super().__init__()
        self.n_max = n_max

    def __call__(self, data):
        n = torch.tensor([data.num_nodes]).to(data.edge_index.device)
        eigfeat, eigvals = self.eigen_features(data.edge_index, data.edge_attr, data.batch, n)
        data.eigen_features = eigfeat
        return data

    def eigen_features(self, edge_index, edge_attr, batch, n):
        # if edge_attr is not None:
        #     if edge_attr.shape[-1] > 1:
        #         edge_index = edge_index[:, edge_attr[..., :-1].sum(-1) > 0]
        #     else:
        #         edge_index = edge_index[:, edge_attr.sum(-1) > 0]
        lap, weight = get_laplacian(edge_index, edge_weight=edge_attr.argmax(dim=-1).float())
        lap = to_dense_adj(lap, edge_attr=weight, batch=batch).squeeze()
        eigvals, eigvectors = torch.linalg.eigh(lap)
        is_zero = torch.round(eigvals, decimals=6) != 0
        eigvectors = eigvectors * is_zero.unsqueeze(-2)
        #eigvectors = eigvectors * n.sqrt().view(-1, 1, 1)
        eigvectors.squeeze_()
        K = self.n_max
        eigfeat = eigvectors[..., 1: K + 1]
        if eigfeat.size(-1) < K:
            d = K - eigfeat.size(-1)
            n = eigfeat.size(0)
            eigfeat = torch.cat((eigfeat, torch.zeros((n, d), device=eigfeat.device)), dim=-1)
        return eigfeat, eigvals
