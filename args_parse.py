import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='zinc',
        help="Name of the dataset. Available:  zinc, enzymes, proteins"
    )

    parser.add_argument(
        "--wandb", type=str,
        default='disabled', help="If W&B is online, offline or disabled"
    )

    return parser.parse_args()
