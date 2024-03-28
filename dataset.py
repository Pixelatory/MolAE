from functools import partial
from typing import Dict, List, Union
import re
from rdkit import Chem
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from pyspark.sql import SparkSession
import pyarrow.parquet as pq
from tqdm import tqdm
import multiprocessing as mp
import random
try:
    import selfies as sf
except ImportError:
    pass

def load_vocab_from_file(file_name: str):
    with open(file_name, 'r') as f:
        return [line.strip() for line in f.readlines()]
    

def get_encoders(vocab: List[str], additional_model_tokens: List[str]=None):
    """
        :param additional_model_tokens:
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {}
    for token in additional_model_tokens:
        model_str2num[token] = len(model_str2num)

    vocab_str2num = {}
    for token in vocab:
        if token in model_str2num.keys():
            raise Exception(f"Overlap in tokens between vocabulary and model token: {token}")
        vocab_str2num[token] = len(vocab_str2num) + len(model_str2num)

    return model_str2num, vocab_str2num


def get_decoders(*args):
    return [{i: j for j, i in entry.items()} for entry in args]


def tokenize_molecule(smi, use_selfies=False, pattern=None):
    if use_selfies:
        return sf.split_selfies(sf.encoder(smi))
    if pattern is None:
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0-9])"  # default regex pattern for SMILES strings
    return re.findall(pattern, smi)


def encode_token(token, model_str2num, vocab_str2num, unk_token="[UNK]"):
    if token in model_str2num:
        return model_str2num[token]
    elif token in vocab_str2num:
        return vocab_str2num[token]
    
    return model_str2num[unk_token]


def encode_seq(seq: str, model_str2num: Dict[str, int], vocab_str2num: Dict[str, int], prepend_tokens: List[str] = None, 
               append_tokens: List[str] = None, unk_token="[UNK]", **kwargs) -> List[int]:
    """
        Encodes sequence using SMILES/SELFIES vocabulary. If token not in either vocabs, use model vocab's [UNK] token.
    """
    if prepend_tokens is None:
        prepend_tokens = []
    if append_tokens is None:
        append_tokens = []
    encoded_prepend = [encode_token(token, model_str2num, vocab_str2num, unk_token) for token in prepend_tokens]
    encoded_append = [encode_token(token, model_str2num, vocab_str2num, unk_token) for token in append_tokens]
    encoded_seq = [encode_token(token, model_str2num, vocab_str2num, unk_token) for token in tokenize_molecule(seq, kwargs)]
    return encoded_prepend + encoded_seq + encoded_append


def randomize_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles


def canonical_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles


class SmilesDataset(Dataset):
    def __init__(self, dataset_path, smiles_col: str, tasks_cols: Union[str, List[str]], model_str2num, smiles_str2num, use_dask=False):
        cols = []
        if type(tasks_cols) is list:
            cols = tasks_cols
            cols.append(smiles_col)
        else:
            cols = [smiles_col, tasks_cols]
        
        if use_dask:
            df = dd.read_csv(dataset_path, usecols=cols)
        else:
            df = pd.read_csv(dataset_path, usecols=cols)
        

        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num

        if not path.endswith('pkl'):
            self.data = self.df[Smiles_head].to_numpy().reshape(-1).tolist()
            self.data = [self._char_to_idx(entry) for entry in tqdm(self.data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smiles = self.data[item]
        x, y, weights = self.numerical_smiles(smiles)
        return x, y, weights

    def numerical_smiles(self, smiles):
        # nums_list = self._char_to_idx(smiles)
        nums_list = smiles  # smiles is really the encoding
        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = self.model_str2num['[UNK]']
            elif rand < 0.9:
                if self.fragmentation:
                    nums_list[i] = int(
                        np.random.randint(len(self.model_str2num), len(self.smiles_str2num) + len(self.frag_str2num)))
                else:
                    nums_list[i] = int(np.random.randint(len(self.model_str2num), len(self.smiles_str2num)))
        x = np.array(nums_list).astype('int64')
        weights = weight.astype('float32')
        return x, y, weights

    def _char_to_idx(self, seq, prepend=None, append=None):
        encoding = encode_molecule(seq, self.model_str2num, self.smiles_str2num)
        return prepend + encoding + append

    def pickle_data_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.data, f)

class ParquetDataset:
    def __init__(self, dataset_path) -> None:
        self.data = dd.read_parquet(dataset_path)

    def get_partition(self, idx):
        return self.data.get_partition(idx).compute()
    
    def n_partitions(self):
        return self.data.npartitions
    
    def load_partitions(self, idxs):
        return pd.concat([self.get_partition(idx) for idx in idxs])


class ParquetDataLoader:
    def __init__(self, 
                 dataset: ParquetDataset,
                 model_str2num: Dict[str, int],
                 vocab_str2num: Dict[str, int],
                 smiles_col: str = 'smiles', 
                 shuffle: bool = False,
                 batch_size: int = 1,
                 use_selfies: bool = False,
                 mlm_prob_overall: float = 0.15,
                 mlm_prob_token_mask: float = 0.8,
                 mlm_prob_token_random: float = 0.1,
                 mlm_mask_token: str = "[MASK]",
                 n_partitions_loaded: int = 10,
                 random_state: int = 42,
                 model_unk_token: str = "[UNK]",
                 append_tokens_to_seq: Union[str, List[str]] = None,
                 prepend_tokens_to_seq: Union[str, List[str]] = None):
        """
            Note: By default assumes masked language modelling task, however if mlm_prob_overall=0, then assumes language modelling task.
        """
        if mlm_prob_overall > 1.0:
            raise Exception("Masked language modelling token selection probability cannot be > 1.0")
        if mlm_prob_token_mask + mlm_prob_token_random > 1.0:
            raise Exception(f"Probability of MLM token replacement with {mlm_mask_token} + probability \
                            of replacing token with random from vocab cannot be > 1.0")
        if batch_size <= 0:
            raise Exception("Batch size must be > 0")
        if n_partitions_loaded <= 0:
            raise Exception("Number of loaded partitions must be > 0")
        
        # DATASET VARIABLES
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_partitions_loaded = n_partitions_loaded
        self.partition_idxs = [i for i in range(self.dataset.n_partitions())]
        self.smiles_col = smiles_col

        # VOCABULARY & RANDOM GENERATOR VARIABLES
        self.model_str2num = model_str2num
        self.vocab_str2num = vocab_str2num
        self.use_selfies = use_selfies
        self.model_unk_token = model_unk_token
        self.generator = np.random.default_rng(random_state)

        if append_tokens_to_seq is type(str):
            self.append = [append_tokens_to_seq]
        else:
            self.append = append_tokens_to_seq

        if prepend_tokens_to_seq is type(str):
            self.prepend = [prepend_tokens_to_seq]
        else:
            self.prepend = prepend_tokens_to_seq    

        # MASKED LANGUAGE MODELLING VARIABLES
        self.mlm_prob_overall = mlm_prob_overall
        self.mlm_prob_token_mask = mlm_prob_token_mask
        self.mlm_prob_token_replace = mlm_prob_token_random + mlm_prob_token_mask
        self.mlm_mask_token = mlm_mask_token

    def __iter__(self):
        loaded_data = []
        if self.shuffle:
            self.generator.shuffle(self.partition_idxs)

        for i in range(0, len(self.partition_idxs), self.n_partitions_loaded):
            # load up n partitions
            partition_start = i
            partition_end = min(partition_start + self.n_partitions_loaded, len(self.partition_idxs))

            selected_partitions = [self.partition_idxs[i] for i in range(partition_start, partition_end)]
            loaded_partitions = self.dataset.load_partitions(selected_partitions)
            loaded_partitions = loaded_partitions[self.smiles_col].tolist()

            # shuffle SMILES strings in partitions
            if self.shuffle:
                self.generator.shuffle(loaded_partitions)

            loaded_data += loaded_partitions
            for i in range(0, len(loaded_data), self.batch_size):
                batch = loaded_data[i:i+self.batch_size]
                if len(batch) < self.batch_size:
                    # any remaining data that doesn't fit into batch is used at start of next load of partition data
                    loaded_data = batch
                else:
                    batch = [self._encode_seq(smi) for smi in batch]
                    if self.mlm_prob_overall == 0:
                        yield [self._lm_procedure(encoded_seq) for encoded_seq in batch]
                    else:
                        yield [self._mlm_procedure(encoded_seq) for encoded_seq in batch]

    def _encode_seq(self, seq: str):
        return encode_seq(seq, self.model_str2num, self.vocab_str2num, self.prepend, self.append, unk_token=self.model_unk_token, use_selfies=self.use_selfies)
    
    def _mlm_procedure(self, encoded_seq: List[int]):
        """
            Masked language modelling procedure on an encoded sequence.
        """
        masked_tokens = self.generator.permutation(len(encoded_seq) - 1)[:int(len(encoded_seq) * self.mlm_prob_overall)] + 1
        x = torch.tensor(encoded_seq, dtype=torch.int64)
        y = torch.tensor(encoded_seq, dtype=torch.int64)
        weights = torch.zeros(len(encoded_seq), dtype=torch.float32)

        for i in masked_tokens:
            rand = self.generator.random()
            weights[i] = 1.0
            if rand < self.mlm_prob_token_mask:
                x[i] = self.model_str2num[self.mlm_mask_token]
            elif rand < self.mlm_prob_token_replace:
                rand_vocab_idx = (len(self.vocab_str2num) - len(self.model_str2num)) * self.generator.random() + len(self.model_str2num)
                x[i] = int(rand_vocab_idx)

        return x, y, weights
    
    def _lm_procedure(self, encoded_seq: List[int]):
        """
            Language modelling procedure on an encoded sequence.
        """
        x = torch.tensor(encoded_seq, dtype=torch.int64)
        y = torch.tensor(encoded_seq, dtype=torch.int64)
        return x, y


if __name__ == "__main__":
    selfies_vocab = load_vocab_from_file('E:/ZINC-20-all/selfies_vocab.txt')
    model_str2num, selfies_str2num = get_encoders(selfies_vocab, ['[PAD]', '[UNK]', '[MASK]', '[CLS]'])
    dataset = ParquetDataset('E:/ZINC-20-all/parquet')
    dataloader = ParquetDataLoader(dataset, model_str2num, selfies_str2num, shuffle=True, use_selfies=True, n_partitions_loaded=10, batch_size=5, mlm_prob_overall=0.5, prepend_tokens_to_seq=['[CLS]'])

    for batch in tqdm(dataloader):
        print(batch)
    
    exit(1)
    df = dd.read_parquet('E:/ZINC-20-all/parquet')
    df['rand_index'] = np.random.permutation(len(df))
    print('generated rand index')
    df = df.set_index('rand_index')
    print('done')
    exit(1)
    
    dataset = MyDataset('E:/ZINC-20-all/parquet', 'E:/ZINC-20-all/parquet/partition_sizes.pkl')
    #print(dataset.get_batch([0,1,2,1000000, 20000000]))

    dataloader = MyDataLoader(dataset=dataset, batch_size=1000, shuffle=True)

    for batch in dataloader:
        print(len(batch))
    

    '''
    df = dd.read_csv('E:/ZINC-20-all/zinc-20.csv').repartition(partition_size="5MB").to_parquet('E:/ZINC-20-all/parquet')
    partition_indexes = list(range(df.npartitions))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        partition_sizes = list(tqdm(pool.imap(get_partition_size, partition_indexes), total=len(partition_indexes)))
   
    
    with open('E:/ZINC-20-all/parquet/partition_sizes.pkl', 'wb') as f:
        pickle.dump(partition_sizes, f)
    '''

    #vocab = load_vocab_from_file('E:/ZINC-20-all/smiles_vocab.txt')
    #model_str2num, smiles_str2num = get_encoders(vocab, ['[PAD]', '[UNK]', '[MASK]', '[CLS]', '[SEP]'])
    #print(model_str2num)m
    #print(smiles_str2num)