import os
import re
import sys
from typing import List
import pandas as pd
from tdc.utils import retrieve_label_name_list
from tdc.generation import MolGen
from tdc.single_pred import ADME, Tox, HTS, Yields
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
    data = Yields(name = 'Buchwald-Hartwig').get_data().iloc[0]['Reaction']
    print(data)
    exit(1)
    # FINETUNING
    data = ADME(name = 'Caco2_Wang').get_data()
    data = ADME(name = 'PAMPA_NCATS').get_data()
    data = ADME(name = 'PAMPA_NCATS').get_approved_set()
    data = ADME(name = 'HIA_Hou').get_data()
    data = ADME(name = 'Pgp_Broccatelli').get_data()
    data = ADME(name = 'Bioavailability_Ma').get_data()
    data = ADME(name = 'Lipophilicity_AstraZeneca').get_data()
    data = ADME(name = 'Solubility_AqSolDB').get_data()
    data = ADME(name = 'HydrationFreeEnergy_FreeSolv').get_data()
    data = ADME(name = 'BBB_Martins').get_data()
    data = ADME(name = 'PPBR_AZ').get_data()
    data = ADME(name = 'VDss_Lombardo').get_data()
    data = ADME(name = 'CYP2C19_Veith').get_data()
    data = ADME(name = 'CYP2D6_Veith').get_data()
    data = ADME(name = 'CYP3A4_Veith').get_data()
    data = ADME(name = 'CYP1A2_Veith').get_data()
    data = ADME(name = 'CYP2C9_Veith').get_data()
    data = ADME(name = 'CYP2C9_Substrate_CarbonMangels').get_data()
    data = ADME(name = 'CYP2D6_Substrate_CarbonMangels').get_data()
    data = ADME(name = 'CYP3A4_Substrate_CarbonMangels').get_data()
    data = ADME(name = 'Half_Life_Obach').get_data()
    data = ADME(name = 'Clearance_Hepatocyte_AZ').get_data()
    data = ADME(name = 'Clearance_Microsome_AZ').get_data()
    data = Tox(name = 'LD50_Zhu').get_data()
    data = Tox(name = 'hERG').get_data()

    label_list = retrieve_label_name_list('herg_central')
    for label_name in label_list:
        data = Tox(name = 'herg_central', label_name = label_name).get_data()
    
    data = Tox(name = 'hERG_Karim').get_data()
    data = Tox(name = 'AMES').get_data()
    data = Tox(name = 'DILI').get_data()
    data = Tox(name = 'Skin Reaction').get_data()
    data = Tox(name = 'Carcinogens_Lagunin').get_data()

    label_list = retrieve_label_name_list('Tox21')
    for label_name in label_list:
        data = Tox(name = 'Tox21', label_name = label_name).get_data()
    
    label_list = retrieve_label_name_list('Toxcast')

    for label_name in label_list:
        data = Tox(name = 'ToxCast', label_name = label_name).get_data()
    
    data = Tox(name = 'ClinTox').get_data()

    data = HTS(name = 'SARSCoV2_Vitro_Touret').get_data()
    data = HTS(name = 'SARSCoV2_3CLPro_Diamond').get_data()
    data = HTS(name = 'HIV').get_data()
    data = HTS(name = 'orexin1_receptor_butkiewicz').get_data()
    data = HTS(name = 'm1_muscarinic_receptor_agonists_butkiewicz').get_data()
    data = HTS(name = 'm1_muscarinic_receptor_antagonists_butkiewicz').get_data()
    data = HTS(name = 'potassium_ion_channel_kir2.1_butkiewicz').get_data()
    data = HTS(name = 'kcnq2_potassium_channel_butkiewicz').get_data()
    data = HTS(name = 'cav3_t-type_calcium_channels_butkiewicz').get_data()
    data = HTS(name = 'choline_transporter_butkiewicz').get_data()
    data = HTS(name = 'serine_threonine_kinase_33_butkiewicz').get_data()
    data = HTS(name = 'tyrosyl-dna_phosphodiesterase_butkiewicz').get_data()
    exit(1)

    # PRETRAINING
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