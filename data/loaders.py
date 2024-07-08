import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url
from utils.func import atom_number_to_one_hot, from_dense_numpy_to_sparse


class PyGDataset(InMemoryDataset):
    def __init__(self, root, dataset, config, data_info, transform=None,
                 pre_transform=None, pre_filter=None, data_level_up=None, data_info_prev=None):
        self.dataset = dataset
        self.config = config
        self.data_info = data_info
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transform
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        if  self.dataset == 'zinc':
            return ['zinc_kekulized.npz']
        elif self.dataset == 'qm9':
            return ['qm9_kekulized.npz']
        else:
            raise NotImplementedError()

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'zinc':
            download_url('https://drive.switch.ch/index.php/s/D8ilMxpcXNHtVUb/download', self.raw_dir,
                         filename='zinc_kekulized.npz')
        elif self.dataset == 'qm9':
            download_url('https://drive.switch.ch/index.php/s/SESlx1ylQAopXsi/download', self.raw_dir,
                         filename='qm9_kekulized.npz')

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def process(self):
        print(f'Processed files stored in .{os.path.dirname(self.processed_paths[0])}')
        if not os.path.isdir(os.path.dirname(self.processed_paths[0])):
            os.mkdir(os.path.dirname(self.processed_paths[0]))

        data_list = self.process_molecular_dataset()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process_molecular_dataset(self):
        filepath = os.path.join(self.raw_dir, f'{self.dataset}_kekulized.npz')
        load_data = np.load(filepath)
        xs = load_data['arr_0']
        adjs = load_data['arr_1']
        del load_data
        data_list = []
        for i, (x, adj) in enumerate(zip(xs, adjs)):
            x = atom_number_to_one_hot(x, self.dataset)
            edge_index, edge_attr = from_dense_numpy_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            if data.x.shape[0]>1:
                data_list.append(data)
            if (i + 1) % 1000 == 0:
                print(f'{i + 1} graphs processed... process continue')
        return data_list
