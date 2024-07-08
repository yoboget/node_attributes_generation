import yaml
from easydict import EasyDict
import wandb

zinc = {'max_num_nodes': 38, 'n_node_attr': 9, 'n_edge_attr': 3, 'mol': True}
qm9 = {'max_num_nodes': 9, 'n_node_attr': 4, 'n_edge_attr': 3, 'mol': True}

DATA_INFO = {'zinc': zinc, 'qm9': qm9}

def get_config(args):
    # MODEL CONFIG
    config_path = f'./config/{args.dataset}.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    config = EasyDict(config)
    config.dataset = args.dataset
    config.model_dir = wandb.run.dir

    # DATA INFO
    data_info = DATA_INFO[config['dataset']].copy()
    data_info = EasyDict(data_info)
    return config, data_info
