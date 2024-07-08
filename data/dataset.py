import os
import torch
from data.loaders import PyGDataset
from data.transforms import Spectral
from torch_geometric.loader import DataLoader
from easydict import EasyDict
from typing import List, Tuple
from data.utils import get_indices



def get_dataset(config: EasyDict, data_info: EasyDict) -> Tuple[List[DataLoader], EasyDict, EasyDict]:
    # Choose the appropriate transforms based on the dataset and configuration
    dataset = config.dataset
    transforms = Spectral(data_info.max_num_nodes)
    data = PyGDataset(f'./data/{dataset}/', dataset, config=config, data_info=data_info,
                          pre_transform=transforms)

    if dataset == 'zinc' or dataset == 'qm9':
        train_idx, test_idx, test_size = get_indices(config, dataset, len(data))
        VAL_SIZE = 10000
        idx = torch.randperm(len(train_idx))
        train_idx = train_idx[idx]
        train_idx, val_idx = train_idx[VAL_SIZE:], train_idx[:VAL_SIZE]


    train_loader = DataLoader(data[train_idx], batch_size=config.training.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(data[val_idx], batch_size=config.training.val_batch_size,
                             shuffle=False, drop_last=True)
    test_loader = DataLoader(data[test_idx], batch_size=config.training.val_batch_size)
    loaders = train_loader, val_loader, test_loader

    return loaders, config, data_info
