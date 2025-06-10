import utils as U
import glob
import pandas as pd
import polars as pl
from pathlib import Path
import h5py as h5
import numpy as np
from typing import Dict
import warnings


@U.timeit
def prepare_annotations(
    df: pd.DataFrame,
    basic_features: bool = True,
    expression_levels: bool = False,
    include_hpo: bool = True,
    include_go: bool = True,
) -> pd.DataFrame:
    """Given a dataset of variants process feature columns for training"""

    def enc_hpo(df_ann):
        df_ann["HPOTerms"] = df_ann["HPOTerms"].str.strip('"')
        df_oh = df_ann["HPOTerms"].str.get_dummies(sep=",")
        df_ann = pd.concat([df_ann, df_oh], axis=1)

        df_ann["HPOTermParents"] = df_ann["HPOTermParents"].str.strip('"')
        df_oh = df_ann["HPOTermParents"].str.get_dummies(sep=",")
        df_ann = pd.concat([df_ann, df_oh], axis=1)

        return df_ann

    def enc_go(df):
        df["GOTerms"] = df["GOTerms"].str.strip('"')
        df_oh = df["GOTerms"].str.get_dummies(sep=";")
        df = pd.concat([df, df_oh], axis=1)
        return df

    df = df.drop(
        ["substituted_aa", "original_aa", "position", "MIM"], axis=1, errors="ignore"
    )

    if not expression_levels:
        cols = list(df.columns)
        cols = [c for c in cols if c[:4] != "rel_" and c[:4] != "abs_"]
        df = df[cols]

    if not basic_features:
        # Only keep HPO columns
        try:
            df = df[
                [
                    "VarID",
                    "HPOTerms",
                    "HPOTermParents",
                    "GOTerms",
                    "netChange",
                    "gene",
                    "sequence",
                    "train",
                ]
            ]
        except KeyError:
            df = df[
                [
                    "VarID",
                    "HPOTerms",
                    "HPOTermParents",
                    "GOTerms",
                    "netChange",
                    "gene",
                    "sequence",
                ]
            ]

    if include_hpo:
        df = enc_hpo(df)

    if include_go:
        df = enc_go(df)
    df = df.drop(labels=["GOTerms"], axis=1)

    # df = df.groupby(axis=1, level=0).sum()
    df = df.T.groupby(level=0).sum().T

    df = df.drop(labels=["HPOTerms", "HPOTermParents"], axis=1, errors="ignore")
    return df


@U.timeit
def prepare_go_ann(df, include_go: bool = False):
    if include_go:
        df = enc_go(df)

    df = df.drop(labels=["GOTerms"], axis=1)

    return df


def convert_netchange(row):
    if row["netChange"] == "lof":
        row["netChange"] = 0
    elif row["netChange"] == "gof":
        row["netChange"] = 1
    else:
        raise ValueError(f"netChange has unsupported value: {row['netChange']}")
    return row


@U.timeit
def prepare_dataset(
    variants: str,
    annotations: str,
    embedding_dir: str,
    basic_features: bool = True,
    expression_levels: bool = False,
    include_hpo: bool = False,
    include_go: bool = False,
    keep_only_hpo: bool = False,
):
    """
    Loads a dataset and processes columns for training
    """
    df = pd.read_csv(variants, sep="\t")
    df_ann = pd.read_csv(annotations, sep="\t")

    df = df[df["netChange"] != "neutral"][["VarID", "gene", "netChange", "sequence"]]
    print("df:", df.shape)

    df_ann = df_ann[df_ann["VarID"].isin(df["VarID"])]
    print("df_ann:", df_ann.shape)

    df = pd.merge(df, df_ann, on="VarID", suffixes=[None, "_y"])
    df = df.drop(columns=["gene_y"])
    print("df merged:", df.shape)

    emb_keys = get_embedding_keys(embedding_dir)
    # print(set(df['VarID']).difference(emb_keys))
    # assert (
    #    len(set(df["VarID"]).difference(emb_keys)) == 0
    # ), "There are variants missing esm embeddings"

    df["AM"] = df["AM"].fillna(df["AM"].mean())
    df["esm_llr"] = df["esm_llr"].fillna(df["esm_llr"].mean())

    # with pd.option_context("future.no_silent_downcasting", True):
    #    df = df.replace({"netChange": {"lof": 0, "gof": 1}})
    df = df.apply(convert_netchange, 1)

    df = prepare_annotations(
        df, basic_features, expression_levels, include_hpo, include_go
    )

    # df = prepare_go_ann(df, include_go)

    if keep_only_hpo:
        # Drop any varients without HPO terms
        hpo_missing = set(df.loc[df["HPOTerms"] == '"HP:9999998"']["VarID"].to_list())
        df = df[~df["VarID"].isin(hpo_missing)]
        print("Drop missing HPO:", df.shape)

    df = df.reset_index(drop=True)

    return df


@U.timeit
def get_embedding_keys(dir_path: str) -> list:
    if ".hdf5" in dir_path:
        with h5.File(dir_path, "r") as hf:
            return list(hf.keys())
    elif "t33_650M" in dir_path:
        files = glob.glob(f"{dir_path}/*.npy")
        ids = []
        for f in files:
            # varid = Path(f).stem.split("_esm")[0]
            varid = Path(f).stem
            ids.append(varid)
        return ids
    else:
        files = glob.glob(f"{dir_path}/*.safetensors")
        ids = []
        for f in files:
            varid = Path(f).stem.split("_esm")[0]
            ids.append(varid)
        return ids


@U.timeit
def get_gene_sizes(df: pd.DataFrame) -> Dict:
    """Return the length of each gene sequence"""
    unique_genes = list(df["gene"].unique())
    sizes = {}
    """
    for g in unique_genes:
        i = list(df.loc[df["gene"] == g, "VarID"])[0]
        s = embeddings[i].shape[0]
        sizes[g] = s
    """

    # Read the sequences lengths from the overview df
    for g in unique_genes:
        sub = list(df.loc[df["gene"] == g, "sequence"])
        lens = set([len(i) for i in sub])
        assert len(lens) == 1, (
            f"Sequence with various lengths for the same gene: {g}:{lens}"
        )
        sizes[g] = list(lens)[0] + 2  # +2 because of BOS/EOS tokens

    return sizes
