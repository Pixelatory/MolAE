import os
import re
import sys
from typing import List
import pandas as pd
from tdc.generation import MolGen
import selfies as sf
from rdkit import Chem
from tqdm import tqdm

sys.path.append('../MOLMIM')  # CHANGE ME
from utils import SMI_REGEX_PATTERN

def process_dataset(dataset: pd.DataFrame, max_token_len: int, smiles_col: str = 'smiles') -> List[str]:
    smis = set()
    unique_smi_tokens = set()
    unique_sf_tokens = set()
    for smi in tqdm(dataset[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smi)
            for a in mol.GetAtoms():
                a.SetAtomMapNum(0)
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            smi_tokens = re.split(SMI_REGEX_PATTERN, cano_smi)
            if len(smi_tokens) <= max_token_len:
                unique_sf_tokens.update(sf.split_selfies(sf.encoder(cano_smi)))  # will be an exception if can't be kept as a SELFIES string
                unique_smi_tokens.update(smi_tokens)
                smis.add(cano_smi)
        except Exception as e:
            print(f"Could not process {smi}: {e}")
    
    return list(smis), list(unique_smi_tokens), list(unique_sf_tokens)

if __name__ == "__main__":
    all_smis = set()
    all_smi_tokens = set()
    all_sf_tokens = set()

    data = MolGen(name = 'MOSES').get_data()
    smis, smi_tokens, sf_tokens = process_dataset(data, 255)
    all_smis.update(smis)
    all_smi_tokens.update(smi_tokens)
    all_sf_tokens.update(sf_tokens)
    os.remove('data/moses.tab')

    data = MolGen(name = 'ChEMBL_V29').get_data()
    smis, smi_tokens, sf_tokens = process_dataset(data, 255)
    all_smis.update(smis)
    all_smi_tokens.update(smi_tokens)
    all_sf_tokens.update(sf_tokens)
    os.remove('data/chembl_v29.csv')

    data = MolGen(name = 'ZINC').get_data()
    smis, smi_tokens, sf_tokens = process_dataset(data, 255)
    all_smis.update(smis)
    all_smi_tokens.update(smi_tokens)
    all_sf_tokens.update(sf_tokens)
    os.remove('data/zinc.tab')

    all_smis = list(all_smis)

    with open('data/allmolgen_255maxlen_cano.csv', 'w') as f:
        f.write('smiles\n')
        for smi in all_smis:
            f.write(f'{smi}\n')
    
    with open('data/smiles_vocab.txt', 'w') as f:
        for smi_token in all_smi_tokens:
            f.write(f"{smi_token}\n")
    
    with open('data/selfies_vocab.txt', 'w') as f:
        for sf_token in all_sf_tokens:
            f.write(f"{sf_token}\n")