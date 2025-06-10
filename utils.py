import numpy as np
import os
import time
import torch
import pandas as pd
import h5py as h5
from sklearn.metrics import (
    matthews_corrcoef,
)
from sklearn.metrics import f1_score as sklearn_f1_score


def merge_on_varid(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge two pd.DataFrame's on the VarID column"""
    return pd.merge(
        left, right, on="VarID", how="outer", suffixes=[None, "_right"], indicator=True
    )


def load_lmdb(key: str, shape: tuple[int], lmdb_dir):
    # Open LMDB environment
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, sync=True, readahead=False)

    # Begin transaction
    with env.begin() as txn:
        # Retrieve data by key
        data_bytes = txn.get(key.encode())

        if data_bytes is None:
            raise KeyError(f"Key '{key}' not found in LMDB database.")

        # Decode data into NumPy array
        numpy_array = np.frombuffer(
            data_bytes, dtype=np.float32
        )  # Adjust dtype if needed

    # Close environment
    env.close()

    return numpy_array.reshape(*shape)


def dataframe_to_dict(df, key_column):
    data_dict = {}
    for row in df.rows(named=True):
        key = row[key_column]
        values = [row[col] for col in df.columns if col != key_column]
        data_dict[key] = values
    return data_dict


def timeit(func):
    """Function timer"""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"> {func.__name__} :: {(time.perf_counter() - start):.3f} sec")
        return out

    return wrapper


def sample_wise_norm(x, eps=1e-8):
    sample_mean = np.mean(x, axis=(1, 2), keepdims=True)
    sample_std = np.std(x, axis=(1, 2), keepdims=True)
    x = (x - sample_mean) / (sample_std + eps)  # Adding epsilon for numerical stability
    return x


def sequence_wise_norm(x, eps=1e-8):
    sequence_mean = np.mean(x, axis=(0, 2), keepdims=True)
    sequence_std = np.std(x, axis=(0, 2), keepdims=True)
    x = (x - sequence_mean) / (sequence_std + eps)
    return x


def feature_wise_norm(x, eps=1e-8):
    # Get the feature mean/std across all samples and residues
    # Each residue will have features with mean 0 with unit variance
    feature_mean = np.mean(x, axis=(0, 1), keepdims=True)
    feature_std = np.std(x, axis=(0, 1), keepdims=True)
    x = (x - feature_mean) / (feature_std + eps)
    return x


class Timer:
    """Context timer"""

    def __init__(self, name, resolution: int = 1):
        self.name = name
        self.resolution = resolution

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print(f"{self.name} :: {(time.time() - self.start):.{self.resolution}f}s")


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


@timeit
def load_esm(genes: list[dict], esm_root: str, esm_model_name: str):
    embeddings = []
    len_ = len(genes)
    for idx, g in enumerate(genes[:len_]):
        gene = g["gene"]
        aa = g["original_aa"]
        pos = g["position"]
        bb = g["substituted_aa"]
        gene_aa_pos_bb = f"{gene}_{aa}_{pos}_{bb}"
        embeddings.append(
            np.load(
                f"{esm_root}/{esm_model_name}/{gene_aa_pos_bb}_{esm_model_name}.npz"
            )["a"][0]
        )

    embeddings = torch.nested.nested_tensor(embeddings)  # (n_samples, ragged, 1280)
    print(
        "emb.shape():",
        embeddings.size(0),
        "<Ragged>",
        embeddings.size(2),
    )

    embeddings = torch.nested.to_padded_tensor(
        embeddings,
        padding=0.0,
        output_size=(len_, 2018, 1280),
    )

    return np.array(embeddings)


@timeit
def load_esm_hdf(genes: list[dict], hdf_file_path: str):
    embeddings = []
    with h5.File(hdf_file_path, "r") as hf:
        for idx, g in enumerate(genes):
            gene = g["gene"]
            aa = g["original_aa"]
            pos = g["position"]
            bb = g["substituted_aa"]
            name = f"{gene}_{aa}_{int(pos)}_{bb}"
            dset = np.array(hf[name]).squeeze()
            embeddings.append(dset)

    embeddings = torch.nested.nested_tensor(embeddings)  # (n_samples, ragged, 1280)
    print(
        "emb.shape():",
        embeddings.size(0),
        "<Ragged>",
        embeddings.size(2),
    )

    embeddings = torch.nested.to_padded_tensor(
        embeddings,
        padding=0.0,
        output_size=(len(genes), 2018, 1280),
    )

    return np.array(embeddings)


def f1_score(y_true: torch.tensor, y_preds: torch.tensor, average=None):
    """Takes the one-hot labels and raw logit predictions to compute f1-score"""

    y_p = torch.softmax(y_preds, 1)
    y_p = torch.argmax(y_p, 1)
    y_t = torch.argmax(y_true, 1)

    f1 = sklearn_f1_score(
        y_t, y_p, average=average, labels=[0, 1], zero_division=np.nan
    )
    try:
        f1 = [round(i, 2) for i in f1]
    except TypeError:
        f1 = round(f1, 2)
    return f1


def phi_coefficient(y_true: torch.tensor, y_preds: torch.tensor):
    """Matthews correlation coefficient"""
    y_true = torch.argmax(y_true, 1)
    y_pred = torch.argmax(torch.softmax(y_preds, 1), 1)
    return matthews_corrcoef(y_true, y_pred)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory_usage(device=None, value="bytes"):
    if value == "Mb":
        return torch.cuda.memory_allocated(device=device) * 1e-6
    else:
        return torch.cuda.memory_allocated(device=device)
