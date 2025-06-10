"""
Split dataset into train/test for K folds


> This script was written with Arto_Annotations@13b817f
"""

import datetime
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    RepeatedStratifiedKFold,
)
import random
import numpy as np
import sklearn

seed = random.randrange(2**32 - 1)
rng = np.random.default_rng(seed)
print("seed:", seed)

MODEL = "all_variants_am_over_0.3"
MODEL = "all_variants_with_go2"
MODEL = "pm_variants_2"
aa_commit = "13b817f"

note = f"""
# Pre made data splits of our dataset

seed: {seed}
datetime: {datetime.datetime.now().isoformat()}

Uses Arto_Annotations commit: {aa_commit}

Created with /Preprocessing/split_data.py split_all()
"""

df = pd.read_csv(
    "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.tsv", sep="\t"
)

df_ann = pd.read_csv(
    "/home/sean/Data/Latest_VerifiedVariants_Annotations_GOA.tsv", sep="\t"
)
print("Using alternate Annotations file with GOTerms")


df = df[df["netChange"] != "neutral"][["VarID", "gene", "netChange", "sequence"]]
"""
df = df.drop(
    columns=[
        "Unnamed: 0",
        "PMID",
        "Source",
        "DOI",
        "EmptyDataFields",
        "UniProt",
        "MIM",
        "notes",
        "original_aa",
        "position",
        "substituted_aa",
        "HPOTerms",
        "HPOTermsWithParents",
        "cell_current_loss",
        "expressionChange",
    ]
)
"""
print("df:", df.shape)

df_ann = df_ann[df_ann["VarID"].isin(df["VarID"])]

df = pd.merge(df, df_ann, on="VarID", suffixes=[None, "_y"])
df = df.drop(columns=["gene_y", "MIM"])
print("merged df:", df.shape)

df["AM"] = df["AM"].fillna(df["AM"].mean())
df["esm_llr"] = df["esm_llr"].fillna(df["esm_llr"].mean())

DROP_MISSING_HPO = True


def drop_expression_level_columns(df):
    print(">> Dropping expression level columns <<")
    cols = list(df.columns)
    # cols = [c for c in cols if 'rel_' not in c and 'abs_' not in c]
    cols = [c for c in cols if c[:4] != "rel_" and c[:4] != "abs_"]
    return df[cols]


def filter_low_am(df):
    print("filtering out low AM variants")
    df = df.loc[df["AM"] > 0.3]
    df = df.reset_index(drop=True)
    return df


def split_scn5a(df):
    df = df.loc[df["gene"] == "SCN5A"].reset_index(drop=True)
    print(df.shape)

    kf = StratifiedKFold(n_splits=10)
    y = [0 if i == "lof" else 1 for i in df["netChange"].to_list()]
    for i, (train_index, test_index) in enumerate(kf.split(df, y=y)):
        print(len(train_index), len(test_index))

        df.loc[train_index, "train"] = 1
        df.loc[test_index, "train"] = 0

        df.to_csv(f"./Data/data_splits/scn5a_only/fold_{i}.csv", index=False, sep="\t")
        # train.to_csv(f"../Data/data_splits/{MODEL}/fold_{i}_train.csv", index=False)
        # test.to_csv(f"../Data/data_splits/{MODEL}/fold_{i}_test.csv", index=False)

    # with open(f"./Data/data_splits/scn5a_only/README.md", "w") as f:
    #    f.write(f"{note}\n")


def split_mave(df):
    """
    Create a dataset where KCNQ4 (MAVE) is the test set and everything else is a train set
    """
    df_mave = pd.read_csv(
        "/home/sean/Arto_Annotations/Tables/MAVE_Zheng2022.csv", sep="\t"
    )
    df_mave_ann = pd.read_csv(
        "/home/sean/Data/MAVE_Zheng2022_annotations_GOA.tsv", sep="\t"
    )
    df_mave = df_mave[["VarID", "gene", "sequence", "netChange"]]
    df_mave = pd.merge(df_mave, df_mave_ann, on="VarID", suffixes=[None, "_y"])
    df_mave = df_mave.loc[df_mave["netChange"] != "neutral"]

    df = df.loc[df["gene"] != "KCNQ4"]
    print("drop KCNQ4 already in df:", df.shape)

    mave_cols = list(df_mave.columns)
    cols = list(df.columns)
    cols = [c for c in cols if c in mave_cols]
    df = df[cols]
    df_mave = df_mave[cols]

    df = pd.concat([df, df_mave])
    df = df.reset_index(drop=True)
    print(df.shape)

    df["train"] = 1
    df.loc[df["gene"] == "KCNQ4", "train"] = 0

    print(df.head())
    print(df.tail())

    s = df.loc[df["train"] == 1]
    print(s.shape)

    df.to_csv(
        "./Data/data_splits/all_variants_mave_test/fold_0.csv", index=False, sep="\t"
    )


def split_all(df):
    print(df.shape)
    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)

    MODEL = "repeated_k_fold_pm"

    # df = filter_low_am(df)
    # print("AM>0.3:", df.shape)

    if DROP_MISSING_HPO:
        hpo_missing = set(df.loc[df["HPOTerms"] == '"HP:9999998"']["VarID"].to_list())
        df = df[~df["VarID"].isin(hpo_missing)]
        print("df drop missing HPO:", df.shape)
        df = df.reset_index(drop=True)

    # kf = KFold(n_splits=10, shuffle=False, random_state=None)
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=99)
    y = [0 if i == "lof" else 1 for i in df["netChange"].to_list()]
    for i, (train_index, test_index) in enumerate(kf.split(df, y=y)):
        print(len(train_index), len(test_index))

        df.loc[train_index, "train"] = 1
        df.loc[test_index, "train"] = 0

        df.to_csv(f"./Data/data_splits/{MODEL}/fold_{i}.csv", index=False, sep="\t")
        # train.to_csv(f"../Data/data_splits/{MODEL}/fold_{i}_train.csv", index=False)
        # test.to_csv(f"../Data/data_splits/{MODEL}/fold_{i}_test.csv", index=False)

    with open(f"./Data/data_splits/{MODEL}/README.md", "w") as f:
        f.write(f"{note}\n")

    return


def group_fold_split(df):
    print("--------group_fold_split()---------")
    print(df.shape)
    print(df.head())

    df = df.reset_index(drop=True)

    y = [0 if i == "lof" else 1 for i in df["netChange"].to_list()]

    MODEL = "group_fold"

    groups = np.array(df["gene"])
    n_groups = len(set(groups))
    print("n_groups:", n_groups)
    kf = GroupKFold(n_splits=n_groups)
    for i, (train_index, test_index) in enumerate(kf.split(df, y, groups)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={groups[train_index]}")
        print(f"  Test:  index={test_index}, group={groups[test_index]}")
        df.loc[train_index, "train"] = 1
        df.loc[test_index, "train"] = 0
        # df.loc[train_index, "train"] = 1
        # df.loc[test_index, 'train'] = 0

        df.to_csv(f"./Data/data_splits/{MODEL}/fold_{i}.csv", index=False, sep="\t")


def split_pm_bf(df):
    print(df.shape)
    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)

    def add_train(row):
        if len(row["HPOTerms"]) > 15:
            row["train"] = 1
        else:
            row["train"] = 0
        return row

    df = df.apply(add_train, 1)

    df.to_csv("./Data/data_splits/pm_train_bf_test/fold_0.csv", index=False, sep="\t")

    note = """This split sets the variants which have HPO Terms as part of the trainset
    while variants without HPO Terms are part of the test set.\n
"""
    with open("./Data/data_splits/pm_train_bf_test/README.md", "w") as f:
        f.write(note)


def split_funCIon(df):
    training = pd.read_csv("/home/sean/git/funNCion/training_data.csv")
    testing = pd.read_csv("/home/sean/git/funNCion/testing_data.csv")

    train_varids = []
    for idx, row in training.iterrows():
        gene, pos, aa, bb = row["protid"].split(":")
        train_varids.append(f"{gene.upper()}_{aa}_{pos}_{bb}")

    test_varids = []
    for idx, row in testing.iterrows():
        gene, pos, aa, bb = row["protid"].split(":")
        test_varids.append(f"{gene.upper()}_{aa}_{pos}_{bb}")

    df_train = df.loc[df["VarID"].isin(train_varids)]
    print(df_train.shape)

    df_test = df.loc[df["VarID"].isin(test_varids)]
    print(df_test.shape)

    df_train["train"] = 1
    df_test["train"] = 0

    df_final = pd.concat((df_train, df_test))
    print(df_final.shape)

    df_final.to_csv("./Data/data_splits/funCIon/fold_0.csv", index=False, sep="\t")


if __name__ == "__main__":
    df = drop_expression_level_columns(df)

    split_scn5a(df)
    # split_all(df)
    # split_funCIon(df)
    # split_pm_bf(df)
    # split_mave(df)
    # group_fold_split(df)
