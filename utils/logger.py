import time
import os
import wandb
import yaml
import torch
from torchmetrics import MeanMetric, MaxMetric
from easydict import EasyDict as edict
from torch_geometric.utils import to_dense_adj, to_dense_batch, scatter
from utils.mol_utils import gen_mol
from utils.eval import mol_metric
from torch.distributions.categorical import Categorical


class RunningMetric:
    def __init__(self, metric_names: list):
        self.metric_name = metric_names
        self.metrics = {'train': {}, 'iter': {}, 'val': {}}
        for metric_step in self.metrics:
            for metric in metric_names:
                self.metrics[metric_step][metric] = MeanMetric()

    def log(self, step, key, times=None):
        metrics = self.metrics[key]
        log_metrics = {name: metric.compute() for name, metric in metrics.items()}
        if times is not None:
            clock_time = time.time() - times[0]
            process_time = time.process_time() - times[1]
            log_metrics['clock_time'] = clock_time
            log_metrics['process_time'] = process_time
            log_metrics['step'] = step
            print(f'Running metrics for {key} after {step} steps/epochs and {clock_time} seconds')
        else:
            print(f'Running metrics for {key} after {step} steps/epochs')
        print(log_metrics)

        wandb.log({f'{key}': log_metrics})
        for metric in metrics.values():
            metric.reset()
        return log_metrics

    def step(self, to_log, train):
        train_metrics, iter_metrics, val_metrics = self.metrics.values()
        if train:
            for metric, values in zip(train_metrics.values(), to_log):
                metric.update(values)
            for metric, values in zip(iter_metrics.values(), to_log):
                metric.update(values)
        else:
            for metric, values in zip(val_metrics.values(), to_log):
                metric.update(values)


class SamplingMetric:
    def __init__(self, config, data_info):
        self.dataset = config.dataset
        self.model_dir = config.model_dir
        self.n_node_attr = data_info.n_node_attr
        self.n_edge_attr = data_info.n_edge_attr
        self.max_num_nodes = data_info.max_num_nodes
        self.batch_size = config.training.val_batch_size

    def __call__(self, batch, model, epoch):
        input_ = scatter(batch.edge_attr, batch.edge_index[0])
        x_hat = model(input_).softmax(-1)
        x_hat = torch.round(x_hat, decimals=3)

        sampled = Categorical(probs=x_hat).sample().to(x_hat.device)
        x_hat = torch.nn.functional.one_hot(sampled, num_classes=x_hat.shape[1]).to(x_hat.device)

        annots, _ = to_dense_batch(x_hat, batch.batch, max_num_nodes=self.max_num_nodes)
        adjs = to_dense_adj(batch.edge_index, edge_attr=batch.edge_attr,
                            batch=batch.batch, max_num_nodes=self.max_num_nodes)
        adjs = torch.cat((adjs, ((adjs.sum(-1) == 0)*1.).unsqueeze(-1)), dim=-1)

        metrics = self.get_mol_metrics(annots, adjs.permute(0, 3, 1, 2), self.dataset)
        print(metrics)
        wandb.log({f'Sampling Metrics': metrics, 'Sampling epoch': epoch})
        return metrics

    def get_mol_metrics(self, annots, adjs, dataset):
        gen_mols, num_no_correct = gen_mol(annots, adjs, dataset)
        metrics = mol_metric(gen_mols, dataset, num_no_correct, test_metrics=True)
        return metrics


def init_wandb(args):
    wandb.init(project=f'AtomAttr_{args.dataset}',
               config=args, mode=args.wandb)
    wandb.save("*.pt")
    model_dir = wandb.run.dir
    config = vars(args)
    config['model_dir'] = model_dir
    save_name = os.path.join(config['model_dir'], 'my_config.yaml')
    yaml.dump(config, open(save_name, 'w'), Dumper=yaml.Dumper)
    config = edict(config)
    return config
