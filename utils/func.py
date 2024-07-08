
import torch
from torch_geometric.utils import dense_to_sparse


def atom_number_to_one_hot(x, dataset):
    x = x[x > 0]
    if dataset == 'zinc':
        zinc250k_atomic_index = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 3, 4,
                                              0, 0, 0, 0, 0, 5, 6, 7, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 8, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 9])
        x = zinc250k_atomic_index[x] - 1
        x = torch.eye(9)[x]
    else:
        x = torch.eye(4)[x - 6]
    return x

def from_dense_numpy_to_sparse(adj):
    adj = torch.from_numpy(adj)
    no_edge = 1 - adj.sum(0, keepdim=True)
    adj = torch.cat((no_edge, adj), dim=0)
    adj = adj.argmax(0)
    edge_index, edge_attr = dense_to_sparse(adj)
    edge_attr = torch.eye(3)[edge_attr - 1]
    return edge_index, edge_attr

