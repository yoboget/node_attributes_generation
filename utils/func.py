
import torch
from torch_geometric.transforms import LargestConnectedComponents
lcc = LargestConnectedComponents()


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

