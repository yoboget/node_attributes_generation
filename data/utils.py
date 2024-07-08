import os
import json
import torch


def get_indices(config, dataset, n_instances):
    data_dir = './data'
    with open(os.path.join(data_dir, f'valid_idx_{config.dataset}.json')) as f:
        test_idx = json.load(f)
        if dataset == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # Create a boolean mask for the training indices
    train_idx = torch.ones(n_instances).bool()
    train_idx[test_idx] = False
    train_idx = train_idx[train_idx]

    return train_idx, test_idx, len(test_idx)
