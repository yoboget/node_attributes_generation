
import args_parse
from trainer import Trainer
from data.dataset import get_dataset
from config.config import get_config
from utils.logger import init_wandb


def main() -> None:
    # Parse command line arguments
    args = args_parse.parse_args()
    init_wandb(args)

    config, data_info = get_config(args)
    loaders, config, data_info = get_dataset(config, data_info)
    trainer = Trainer(loaders, config, data_info)
    trainer.train()

if __name__ == "__main__":
    main()

