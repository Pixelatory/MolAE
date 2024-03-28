import pandas as pd
from tqdm import tqdm
import re
from rdkit import Chem
import multiprocessing as mp
import selfies as sf
import time

SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0-9])"

active_processes = 0
def dec_active_processes(*args):
    global active_processes
    active_processes -= 1

def build_selfies_vocab(data_file_path, smi_col='smiles', chunksize=None, multiprocess=False, n_processes=None):
    global active_processes
    if chunksize is None and multiprocess is True:
        print("Chunksize is None and multiprocessing is true. Defaulting to single processing.")
    df = pd.read_csv(data_file_path, chunksize=chunksize)

    if multiprocess and chunksize is not None:
        # Multiprocessing
        pool = mp.Pool(processes=n_processes)
        results = []
        for chunk in tqdm(df):
            while active_processes >= n_processes:
                time.sleep(1)
            result = pool.apply_async(process_chunk, args=(chunk, smi_col, False), callback=dec_active_processes)
            active_processes += 1
            results.append(result)
        pool.close()
        pool.join()

        # Merge sets
        unique_tokens = merge_sets([result.get() for result in results])
    else:
        unique_tokens = set()
        if chunksize is not None:
            for chunk in tqdm(df):
                unique_tokens.update(process_chunk(chunk, smi_col, False))
        else:
            for smi in tqdm(df[smi_col]):
                for token in smi_to_sf_tokens(smi):
                    unique_tokens.add(token)
    
    return list(unique_tokens)


def build_smiles_vocab(data_file_path, smi_col='smiles', chunksize=None, multiprocess=False, n_processes=None):
    if chunksize is None and multiprocess is True:
        print("Chunksize is None and multiprocessing is true. Defaulting to single processing.")
    df = pd.read_csv(data_file_path, chunksize=chunksize)

    if multiprocess and chunksize is not None:
        # Multiprocessing
        pool = mp.Pool(processes=None)
        results = []
        for chunk in tqdm(df):
            result = pool.apply_async(process_chunk, args=(chunk, smi_col,))
            results.append(result)
        pool.close()
        pool.join()

        # Merge sets
        unique_tokens = merge_sets([result.get() for result in results])
    else:
        unique_tokens = set()
        if chunksize is not None:
            for chunk in tqdm(df):
                unique_tokens.update(process_chunk(chunk, smi_col))
        else:
            for smi in tqdm(df[smi_col]):
                for token in smi_to_tokens(smi):
                    unique_tokens.add(token)
    
    return list(unique_tokens)

def process_chunk(chunk, smi_col, smiles=True):
    if smiles:
        token_fn = lambda x: smi_to_tokens(x)
    else:
        token_fn = lambda x: smi_to_sf_tokens(x)
    
    unique_tokens = set()
    for smi in chunk[smi_col]:
        for token in token_fn(smi):
            unique_tokens.add(token)
    return unique_tokens

def smi_to_tokens(smi):
    try:
        clean_smi = re.sub(r':[0-9]+', '', smi)  # [CH2:2]1[CH:5]=[CH:9][S:3]/[C:13]1=[C:10](\[OH:11])[NH2:12] -> [CH2]1[CH]=[CH][S]/[C]1=[C](\[OH])[NH2]
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is not None:
            return re.split(SMI_REGEX_PATTERN, clean_smi)
    except:
        pass
    
    return []

def smi_to_sf_tokens(smi):
    try:
        clean_smi = re.sub(r':[0-9]+', '', smi)  # e.g. [CH2:2]1[CH:5]=[CH:9][S:3]/[C:13]1=[C:10](\[OH:11])[NH2:12] -> [CH2]1[CH]=[CH][S]/[C]1=[C](\[OH])[NH2]
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is not None:
            return list(sf.split_selfies(sf.encoder(clean_smi)))
    except:
        pass
    
    return []

def merge_sets(results):
    combined_set = set()
    for result in results:
        combined_set.update(result)
    return combined_set

def save_vocab(vocab, save_path):
    with open(save_path, 'w') as f:
        for token in vocab:
            if token != '':
                f.write(f"{token}\n")

if __name__ == '__main__':
    # CHANGE ME
    smiles_path = 'E:/ZINC-20-all/zinc-20.csv'
    smi_col = 'smiles'
    chunksize = 100000
    multiprocess = True
    n_processes = mp.cpu_count()
    output_selfies_vocab_path = 'E:/ZINC-20-all/selfies_vocab.txt'
    output_smiles_vocab_path = 'E:/ZINC-20-all/smiles_vocab.txt'
    # END OF CHANGE ME

    vocab = build_selfies_vocab(smiles_path, smi_col, chunksize, multiprocess, n_processes)
    save_vocab(vocab, output_selfies_vocab_path)

    vocab = build_smiles_vocab(smiles_path, smi_col, chunksize, multiprocess, n_processes)
    save_vocab(vocab, output_smiles_vocab_path)