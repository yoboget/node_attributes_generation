from utils.chem_func import mols_to_smiles
from graph_stats.stats import eval_graph_list
from utils.mol_utils import mols_to_nx, load_smiles
from fcd_torch import FCD
import rdkit.Chem as Chem
from rdkit import rdBase
import pickle
import torch
import random


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def fraction_unique(gen, k=None, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            raise ValueError(f"Can't compute unique@{k} gen contains only {len(gen)} molecules")
        gen = gen[:k]
    canonic = set(map(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def remove_invalid(gen, canonize=True):
    """
    Removes invalid molecules from the dataset
    """

    if not canonize:
        mols = get_mol(gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in map(canonic_smiles, gen) if x is not None]

def fraction_valid(gen):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
    """
    gen = [mol for mol in map(get_mol, gen)]
    return 1 - gen.count(None) / len(gen)


def novelty(gen, train):
    gen_smiles = []
    for smiles in gen:
        gen_smiles.append(canonic_smiles(smiles))
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def mol_metric(gen_mols, dataset, num_no_correct, test_metrics=False):
    '''
    Args:
        - graphs(list of torch_geometric.Data)
        - train_smiles (list of smiles from the training set)
    Return:
        - Dict with key valid, unique, novel nspdk
    '''
    metrics = {}
    rdBase.DisableLog('rdApp.*')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen_smiles = mols_to_smiles(gen_mols)
    metrics['valid'] = num_no_correct
    gen_valid = remove_invalid(gen_smiles)
    metrics['unique'] = fraction_unique(gen_valid, k=None, check_validity=True)

    if test_metrics:
        train_smiles, test_smiles = load_smiles(dataset=dataset)
        metrics['novel'] = novelty(gen_valid, train_smiles)

        with open(f'./data/{dataset.lower()}_test_nx.pkl', 'rb') as f:
            test_graph_list = pickle.load(f)
            random.Random(42).shuffle(test_graph_list)
        metrics['nspdk'] = eval_graph_list(test_graph_list[:10000], mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

        metrics['fcd'] = FCD(n_jobs=0, device=device)(ref=test_smiles, gen=gen_smiles)
        metrics['valid_with_corr'] = len(gen_valid)
    return metrics


