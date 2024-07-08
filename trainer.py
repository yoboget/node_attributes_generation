
from typing import Tuple
from torch_geometric.utils import scatter
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils.logger import RunningMetric, SamplingMetric


class Trainer:
    def __init__(self, dataloaders, config, data_info):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f'Run on {self.device}')
        self.train_loader, self.val_loader, self.test_loader = dataloaders

        in_ = data_info.n_edge_attr
        out_ = data_info.n_node_attr
        hidden_ = [128, 128, 128]
        self.model = Mlp(in_, out_, hidden_).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.epochs = config.training.epochs
        self.dataset = config.dataset

        # Define Logger
        self.metrics = RunningMetric(['loss'])
        self.n_logging_epochs = config.log.n_loggin_epochs
        self.sampling_metrics = SamplingMetric(config, data_info)

    def train(self) -> None:
        print(f'The training set contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        starting_time = time.time(), time.process_time()
        eval_time = 0

        step = 0
        print('Training starts...')
        for epoch in range(1, self.epochs + 1):
            # TRAIN
            for batch in self.train_loader:
                self.fit(batch.to(self.device))
            self.metrics.log(step, key='train', times=starting_time)

            # TEST
            if epoch % self.n_logging_epochs == 0:
                with torch.no_grad():
                    for batch in self.val_loader:
                        self.fit(batch.to(self.device), train=False)
                    metrics = self.metrics.log(step, key='val', times=starting_time)
                    val_loss = metrics['loss']

                    # EVAL
                    start_eval = time.time()
                    metrics = self.sampling_metrics(batch, self.model, epoch)



    def fit(self, batch: Batch, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model mode (train/eval) based on the "train" parameter
        if train:
            self.opt.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        input_ = scatter(batch.edge_attr, batch.edge_index[0])
        x_hat = self.model(input_)
        loss = F.cross_entropy(x_hat, batch.x)

        if train:
            loss.backward()
            self.opt.step()

        self.metrics.step([loss.item()], train)

class Mlp(nn.Module):
    def __init__(self,
                 in_,
                 out_,
                 hidden_,
                 activation=nn.ReLU()
                 ):
        super().__init__()
        n_layers = len(hidden_) - 1

        layers = [nn.Linear(in_, hidden_[0]), activation]
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_[i], hidden_[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_[-1], out_))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)