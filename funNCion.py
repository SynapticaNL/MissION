import sys
from collections import defaultdict

from train import Archive
import pandas as pd
import numpy as np
from scipy.ndimage import convolve1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

# Joint Nav + Cav alignment file generated with MUSCLE
scn_afa_path = (
    "/home/sean/git/MissION_Annotations/Alt/MUSCLE/AlignedGroups/scn_cacn_algined.fasta"
)
# Joint Potassium genes alignment generated using mafft
kcn_afa_path = "/home/sean/git/MissION_Annotations/Alt/MAFFT/kcn_hcn.mafft.afa"


def ma(x, windowsize):
    """Circular moving average."""
    return convolve1d(x, np.ones(windowsize) / windowsize, mode="wrap")


def split_data(df: pd.DataFrame, seed: int = 999):
    """
    Split dataset into 90/10
    Further split the training fold into 50/50
    """
    training_all, testing = train_test_split(
        df, test_size=0.1, random_state=seed, stratify=df["netChange"]
    )
    training1, training2 = train_test_split(
        training_all,
        test_size=0.5,
        random_state=seed,
        stratify=training_all["netChange"],
    )

    return training1, training2, testing


def get_clean_feature_table(feature_table):
    """Cleans the feature table by dropping columns and duplicates."""
    feature_table = feature_table.copy()
    # Format feature_table
    cols_to_drop = [
        "chr",
        "genomic_pos",
        "USED_REF",
        "STRAND",
        "Feature",
        "inpp2",
        "H",
        "caccon",
        "SF_DEKA",
    ]  # "S2M2",
    feature_table = feature_table.drop(
        columns=[col for col in cols_to_drop if col in feature_table.columns]
    )
    dens_cols = [col for col in feature_table.columns if "dens" in col]
    feature_table = feature_table.drop(columns=dens_cols)
    feature_table = feature_table.drop_duplicates()

    # Crucial fix: The feature_table has duplicate protids. We must enforce a one-to-one relationship.
    feature_table = feature_table.drop_duplicates(
        subset=["VarID"], keep="first"
    )  # drops about 30k rows
    return feature_table


def calculate_and_apply_densities_old(
    training1, training2, testing, feature_table, scn_alignment, kcn_alignment
):
    """Computes variant densities per gene per class lof/gof
    using training1

    Applies the computed values to training2 and testing
    """
    lof_genes = training1[training1["netChange"] == 0]["gene"].unique()
    gof_genes = training1[training1["netChange"] == 1]["gene"].unique()

    gof_scn_family = []
    gof_kcn_family = []
    for i, gene in enumerate(gof_genes):
        gene_type = classify_gene_type(gene)
        variants = training1[
            (training1["gene"] == gene) & (training1["netChange"] == 1)
        ]
        if gene_type == "potassium":
            alignment = kcn_alignment
            gof = gene2family_alignment(gene, variants, alignment)
            gof_kcn_family.append(pd.Series(gof, name=f"{gene}_GOF"))
        else:
            alignment = scn_alignment
            gof = gene2family_alignment(gene, variants, alignment)
            gof_scn_family.append(pd.Series(gof, name=f"{gene}_GOF"))

    lof_scn_family = []
    lof_kcn_family = []
    for i, gene in enumerate(lof_genes):
        gene_type = classify_gene_type(gene)
        variants = training1[
            (training1["gene"] == gene) & (training1["netChange"] == 1)
        ]
        if gene_type == "potassium":
            alignment = kcn_alignment
            lof = gene2family_alignment(gene, variants, alignment)
            lof_kcn_family.append(pd.Series(lof, name=f"{gene}_LOF"))
        else:
            alignment = scn_alignment
            lof = gene2family_alignment(gene, variants, alignment)
            lof_scn_family.append(pd.Series(lof, name=f"{gene}_LOF"))

    family_aligned_scn = pd.concat(gof_scn_family + lof_scn_family, axis=1)
    family_aligned_kcn = pd.concat(gof_kcn_family + lof_kcn_family, axis=1)

    ## Compute VarDensity feature for each alignment family separately and then stack the results
    familyaligned_s = [family_aligned_scn, family_aligned_kcn]
    fams = [scn_alignment, kcn_alignment]

    all = []
    for fam, familyaligned in zip(fams, familyaligned_s):
        # uniqgenemech = feature_table_with_dens['gene'].unique()
        uniqgenemech = [record.name for record in fam]
        feature_table_with_dens = feature_table.loc[
            feature_table["gene"].isin(uniqgenemech)
        ].copy()

        densgof, densgof3aa, denslof, denslof3aa = [], [], [], []

        for gene in uniqgenemech:
            ft_gene = feature_table_with_dens[feature_table_with_dens["gene"] == gene]
            densgof.append(vardens(gene, "_GOF", ft_gene, 10, fam, familyaligned))
            densgof3aa.append(vardens(gene, "_GOF", ft_gene, 3, fam, familyaligned))
            denslof.append(vardens(gene, "_LOF", ft_gene, 10, fam, familyaligned))
            denslof3aa.append(vardens(gene, "_LOF", ft_gene, 3, fam, familyaligned))

        feature_table_with_dens["densgof"] = np.concatenate(densgof)
        feature_table_with_dens["densgof3aa"] = np.concatenate(densgof3aa)
        feature_table_with_dens["denslof"] = np.concatenate(denslof)
        feature_table_with_dens["denslof3aa"] = np.concatenate(denslof3aa)
        all.append(feature_table_with_dens)

    feature_table_with_dens = pd.concat(all)
    # df_all[['VarID', 'densgof', 'densgof3aa', 'denslof', 'denslof3aa']].to_csv('./tmp_vardens.csv', index=False)

    scaler = StandardScaler()
    density_cols = ["densgof", "densgof3aa", "denslof", "denslof3aa"]
    feature_table_with_dens[density_cols] = scaler.fit_transform(
        feature_table_with_dens[density_cols]
    )
    feature_table_with_dens[density_cols] = feature_table_with_dens[density_cols].round(
        2
    )

    # Map variant densities onto training2 and testing data
    density_features = feature_table_with_dens[["VarID"] + density_cols]
    print("density_features:", density_features)

    training2_processed = pd.merge(training2, density_features, on="VarID", how="left")
    print("training2_processed:", training2_processed)
    testing_processed = pd.merge(testing, density_features, on="VarID", how="left")
    print("testing_processed:", testing_processed)

    return training2_processed, testing_processed


def classify_gene_type(gene_name: str):
    if (
        gene_name.startswith("KCN")
        or gene_name.upper() == "SH"
        or gene_name.startswith("HCN")
    ):
        return "potassium"
    else:
        return "sodium_calcium"


def get_gene_alignment(gene: str, alignment: MultipleSeqAlignment) -> np.ndarray:
    for record in alignment:
        if record.name == gene:
            return np.array(list(record.seq))
    raise ValueError(f"Did not find gene={gene} in alignment file")


def gene2family_alignment(
    gene: str, variants: list[str], alignment: MultipleSeqAlignment
):
    variant_counts = variants["position"].value_counts().reset_index()
    variant_counts.columns = ["variant", "Freq"]

    # positions = list(variants['position'])
    # print('positions:', positions)

    gene_alignment = get_gene_alignment(gene, alignment)

    valid_indices = np.where(gene_alignment != "-")[0]

    # By indexing into valid indeces we effectively count ahead the invalid ones
    mapped_indices = valid_indices[variant_counts["variant"] - 1]

    bigfamilyalignment = np.zeros(alignment.get_alignment_length())
    bigfamilyalignment[mapped_indices] = variant_counts["Freq"].values
    return bigfamilyalignment


def vardens(
    gene, funcycat, feature_table_gene, wind, alignmentfile, varonfamilyalignment
):
    """Calculates variant density for a given gene."""
    # Sum densities for the given functional category (e.g., 'GOF')
    densgof = varonfamilyalignment.filter(regex=funcycat).sum(axis=1)

    gene_alignment = get_gene_alignment(gene, alignmentfile)

    # Map onto the specific gene's alignment
    # allvarongene = densgof[alignmentfile[gene] != '-']
    allvarongene = densgof[gene_alignment != "-"]

    # Apply moving average
    slwindall = ma(allvarongene, windowsize=wind)

    # Adapt to multiple aa per site by selecting positions from the feature table
    # feature_table_gene['pos'] is 1-based, so subtract 1 for 0-based indexing
    slwindall = slwindall[feature_table_gene["position"] - 1]
    return slwindall


def get_density_map(training_data, feature_table, scn_alignment, kcn_alignment):
    """
    Compute variant densities per gene per class lof/gof using training_data.
    Returns a mapping of VarID to density features and the fitted scaler.
    """
    # Filter training_data to only include variants that exist in feature_table
    training_data = training_data[
        training_data["VarID"].isin(feature_table["VarID"])
    ].copy()

    lof_genes = training_data[training_data["netChange"] == 0]["gene"].unique()
    gof_genes = training_data[training_data["netChange"] == 1]["gene"].unique()

    gof_scn_family = []
    gof_kcn_family = []
    for gene in gof_genes:
        gene_type = classify_gene_type(gene)
        variants = training_data[
            (training_data["gene"] == gene) & (training_data["netChange"] == 1)
        ]
        if gene_type == "potassium":
            alignment = kcn_alignment
            gof = gene2family_alignment(gene, variants, alignment)
            gof_kcn_family.append(pd.Series(gof, name=f"{gene}_GOF"))
        else:
            alignment = scn_alignment
            gof = gene2family_alignment(gene, variants, alignment)
            gof_scn_family.append(pd.Series(gof, name=f"{gene}_GOF"))

    lof_scn_family = []
    lof_kcn_family = []
    for gene in lof_genes:
        gene_type = classify_gene_type(gene)
        variants = training_data[
            (training_data["gene"] == gene) & (training_data["netChange"] == 0)
        ]
        if gene_type == "potassium":
            alignment = kcn_alignment
            lof = gene2family_alignment(gene, variants, alignment)
            lof_kcn_family.append(pd.Series(lof, name=f"{gene}_LOF"))
        else:
            alignment = scn_alignment
            lof = gene2family_alignment(gene, variants, alignment)
            lof_scn_family.append(pd.Series(lof, name=f"{gene}_LOF"))

    family_aligned_scn = (
        pd.concat(gof_scn_family + lof_scn_family, axis=1)
        if (gof_scn_family + lof_scn_family)
        else pd.DataFrame()
    )
    family_aligned_kcn = (
        pd.concat(gof_kcn_family + lof_kcn_family, axis=1)
        if (gof_kcn_family + lof_kcn_family)
        else pd.DataFrame()
    )

    familyaligned_s = [family_aligned_scn, family_aligned_kcn]
    fams = [scn_alignment, kcn_alignment]

    all = []
    for fam, familyaligned in zip(fams, familyaligned_s):
        # Skip if familyaligned is empty
        if familyaligned.empty:
            continue

        uniqgenemech = [record.name for record in fam]
        feature_table_with_dens = feature_table.loc[
            feature_table["gene"].isin(uniqgenemech)
        ].copy()

        densgof, densgof3aa, denslof, denslof3aa = [], [], [], []

        for gene in uniqgenemech:
            ft_gene = feature_table_with_dens[feature_table_with_dens["gene"] == gene]
            if len(ft_gene) > 0:
                densgof.append(vardens(gene, "_GOF", ft_gene, 10, fam, familyaligned))
                densgof3aa.append(vardens(gene, "_GOF", ft_gene, 3, fam, familyaligned))
                denslof.append(vardens(gene, "_LOF", ft_gene, 10, fam, familyaligned))
                denslof3aa.append(vardens(gene, "_LOF", ft_gene, 3, fam, familyaligned))

        if len(densgof) > 0:
            feature_table_with_dens["densgof"] = np.concatenate(densgof)
            feature_table_with_dens["densgof3aa"] = np.concatenate(densgof3aa)
            feature_table_with_dens["denslof"] = np.concatenate(denslof)
            feature_table_with_dens["denslof3aa"] = np.concatenate(denslof3aa)
            all.append(feature_table_with_dens)

    if len(all) == 0:
        # Return empty density features if no data
        empty_df = pd.DataFrame()
        empty_df["VarID"] = []
        empty_df["densgof"] = []
        empty_df["densgof3aa"] = []
        empty_df["denslof"] = []
        empty_df["denslof3aa"] = []
        return empty_df, StandardScaler()

    feature_table_with_dens = pd.concat(all)

    scaler = StandardScaler()
    density_cols = ["densgof", "densgof3aa", "denslof", "denslof3aa"]
    feature_table_with_dens[density_cols] = scaler.fit_transform(
        feature_table_with_dens[density_cols]
    )
    feature_table_with_dens[density_cols] = feature_table_with_dens[density_cols].round(
        2
    )

    density_features = feature_table_with_dens[["VarID"] + density_cols]

    return density_features, scaler


def apply_density_map(data, density_features, scaler=None):
    """
    Apply pre-computed density features to a dataset.
    Density features should already be scaled using the training scaler.
    """
    processed = pd.merge(data, density_features, on="VarID", how="left")
    return processed


def calculate_and_apply_densities(
    training1, training2, testing, feature_table, scn_alignment, kcn_alignment
):
    """
    Computes variant densities per gene per class lof/gof using training1
    Applies the computed values to training2 and testing
    """
    density_features, scaler = get_density_map(
        training1, feature_table, scn_alignment, kcn_alignment
    )

    # print("density_features:", density_features)

    training2_processed = apply_density_map(training2, density_features)
    # print("training2_processed:", training2_processed)
    testing_processed = apply_density_map(testing, density_features)
    # print("testing_processed:", testing_processed)

    return training2_processed, testing_processed


def encode_categorical_features(training_df: pd.DataFrame, testing_df: pd.DataFrame):
    cols = ["ref_class", "alt_class"]

    train_dummies = pd.get_dummies(training_df[cols])
    test_dummies = pd.get_dummies(testing_df[cols])

    train_dummies, test_dummies = train_dummies.align(
        test_dummies, join="left", axis=1, fill_value=0
    )

    training_df = pd.concat([training_df.drop(columns=cols), train_dummies], axis=1)
    testing_df = pd.concat([testing_df.drop(columns=cols), test_dummies], axis=1)

    return training_df, testing_df


def preprocess(df: pd.DataFrame, scn_alignment, kcn_alignment, seed: int):
    cols_to_drop = [
        "gene",
        "position",
        "original_aa",
        "substituted_aa",
        "caccon",
        "H",
        "UniProtTopo_",
        "ref_class",
        "alt_class",
        "position",
        "sequence",
    ]

    df = df.rename(
        {"original_aa": "refAA", "substituted_aa": "altAA", "position": "pos"}
    )
    # print("df:", df.shape)

    # Change netChange to numeric values
    df["netChange"] = df["netChange"].map({"lof": 0, "gof": 1}).astype("int64")

    feature_table = get_clean_feature_table(df)
    # print("feature_table:", feature_table.shape)

    # Split dataset
    training1, training2, testing = split_data(df, seed=seed)

    training_processed, testing_processed = calculate_and_apply_densities(
        training1, training2, testing, feature_table, scn_alignment, kcn_alignment
    )
    training_processed.to_csv("./tmp_training.csv", index=False)
    # print("feat_with_density:", training_processed.head())

    ## NOTE: I already one-hot encode this feature in the data collection pipeline
    # training_processed, testing_processed = encode_categorical_features(training_processed, testing_processed)

    # Drop excess columns
    training_processed = training_processed.drop(columns=cols_to_drop)
    testing_processed = testing_processed.drop(columns=cols_to_drop)
    c1 = list(training_processed.columns)
    c2 = list(testing_processed.columns)
    assert c1 == c2, "Training and testing columns do not align"

    return training_processed, testing_processed


def fit_model(training: pd.DataFrame, testing: pd.DataFrame):
    """Fit a gradient boosting model on the training/testing data"""

    return


def hyperparameter_search(X_train, y_train, cv=3):
    """
    Run GridSearchCV for GradientBoostingClassifier.

    Args:
        X_train: Feature matrix
        y_train: Target vector
        cv: Number of cross-validation folds

    Returns:
        best_clf: Fitted classifier with best hyperparameters
        best_params: Best hyperparameters found
    """
    param_grid = {
        "n_estimators": [100, 200], #[200, 250, 300, 350, 400],
        "learning_rate": [0.05, 0.1], #[0.05, 0.1, 0.15],
        "subsample": [0.5, 1.0],
        "max_depth": [1, 2, 3], #[1, 3, 5, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3], #[1, 5, 10],
    }

    base_clf = GradientBoostingClassifier(subsample=0.5, random_state=123)

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",#"matthews_corrcoef", #"roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_, grid_search.best_params_


def fit_model_with_tuning(training: pd.DataFrame, testing: pd.DataFrame):
    """Fit a gradient boosting model with hyperparameter tuning."""

    cols_to_drop = [
        "VarID",
        "sequence",
        "netChange",
        "gene",
        "position",
        "original_aa",
        "substituted_aa",
        "ref_class",
        "alt_class",
    ]

    y_train = np.array(training["netChange"])
    X_train = training.drop(columns=cols_to_drop).to_numpy()

    y_test = np.array(testing["netChange"])
    X_test = testing.drop(columns=cols_to_drop).to_numpy()

    si = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    X_train = si.fit_transform(X_train)
    X_test = si.fit_transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    best_clf, best_params = hyperparameter_search(X_tr, y_tr, cv=3)

    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.decision_function(X_test)

    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"Test MCC: {matthews_corrcoef(y_test, y_pred):.3f}")

    return best_clf, best_params


def nested_k_fold(df, scn_alignment, kcn_alignment):
    """Run Nested-K-Fold cross validation"""
    feature_table = get_clean_feature_table(df)

    #archive = Archive("./heyne_impute_scaler_output_gkf.csv", verbose=True)
    archive = Archive("./Out_v2/heyne_impute_scaler_output_gkf.csv", verbose=True)

    # Initial random shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    print("df:", df.shape)

    ## DROP KCN genes
    # df =  df[df["gene"].str.contains(r"(SCN|CACN)", regex=True, na=False)]
    # print('df Nav+Cav:', df.shape)
    
    # 1. OUTER SPLIT: Hold out 10% for final evaluation
    # skf_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #skf_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    ## GROUPED K-FOLD
    groups = np.array(df['gene'])
    n_groups = len(set(groups))
    skf_outer = GroupKFold(n_splits=n_groups)
    fold = 0

    scores = defaultdict(list)
    for train_idx, test_idx in skf_outer.split(df, df["netChange"], groups=groups):
        df_dev = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        g = groups[test_idx][0]
        print(f" --== {fold} ==--")
        print('[test group]:', g)
        # 2. INNER LOOP: Generate features for df_dev using Cross-Fitting
        # This replaces the "training1 / training2" 50/50 split
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        df_dev_with_features = []

        for in_train_idx, in_val_idx in skf_inner.split(df_dev, df_dev["netChange"]):
            inner_train = df_dev.iloc[in_train_idx]
            inner_val = df_dev.iloc[in_val_idx]

            # Fit on 80% of dev, Apply to 20% of dev
            density_map, _ = get_density_map(
                inner_train, feature_table, scn_alignment, kcn_alignment
            )
            val_processed = apply_density_map(inner_val, density_map)
            df_dev_with_features.append(val_processed)

        # Recombine to get the full training set with features
        full_train_processed = pd.concat(df_dev_with_features)
        print("[full_train_processed]:", full_train_processed.shape)

        # 3. OUTER TEST FEATURES: Fit on ALL dev, Apply to test
        final_dev_map, _ = get_density_map(
            df_dev, feature_table, scn_alignment, kcn_alignment
        )
        full_test_processed = apply_density_map(df_test, final_dev_map)
        print('[full_test_processed]:', full_test_processed.shape)

        # full_train_processed.to_csv('./tmp_full_train.csv')

        # drop seq ; seperate target var
        cols_to_drop = [
            "VarID",
            "sequence",
            "netChange",
            "gene",
            "position",
            "original_aa",
            "substituted_aa",
            "ref_class",
            "alt_class",
        ]
        y_train = np.array(full_train_processed["netChange"])
        X_train = full_train_processed.drop(columns=cols_to_drop).to_numpy()

        varid_test = full_test_processed["VarID"]
        y_test = np.array(full_test_processed["netChange"])
        X_test = full_test_processed.drop(columns=cols_to_drop).to_numpy()

        si = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        X_train = si.fit_transform(X_train)
        X_test = si.fit_transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 4. FIT MODEL
        clf = GradientBoostingClassifier(
            n_estimators=200,#100,
            learning_rate=0.1,
            max_depth=3,#1,
            min_samples_leaf=2,#10,
            subsample=0.5,
            random_state=123,
            #learning_rate=0.1,
            #n_estimators=50,
            #max_depth=1,
            #min_samples_leaf=10,
            #random_state=123,
            #subsample=1.0,
        )
        clf.fit(X_train, y_train)
        # score = clf.score(X_train, y_train)
        # print('train score:', score)

        # score = clf.score(X_test, y_test)
        # print('test score:', score)
        # print('------------')

        y_preds = clf.predict(X_test)  # (n_samples,)
        # y_proba = clf.predict_proba(X_test)
        y_proba = clf.decision_function(X_test)
        scores["roc_auc"].append(roc_auc_score(y_test, y_proba[:]))
        scores["ba"].append(balanced_accuracy_score(y_test, y_preds))
        scores["mcc"].append(matthews_corrcoef(y_test, y_preds))

        for i, varid in enumerate(varid_test):
            archive.log(
                {
                    "VarID": varid,
                    "gene": varid.split("_")[0],
                    "mode": "eval",
                    "fold": fold,
                    "epoch": 0,
                    "pred": y_proba[i].tolist(),
                    "label": y_test[i].tolist(),
                }
            )

        fold += 1
        # -- End outer loop --

    for k, v in scores.items():
        print(f"{k} - {np.mean(v):.3f} +- {np.std(v):.4f}")

    archive.flush()

    return


def main(use_hpo = False, use_go = False):
    # Load Alignment files
    scn_afa = AlignIO.read(scn_afa_path, "fasta")
    kcn_afa = AlignIO.read(kcn_afa_path, "fasta")

    # Load feature table
    df = pd.read_csv(
        "/home/sean/git/MissION_Annotations/Alt/Latest_VerifiedVariants_HeyneFeatures.tsv",
        sep="\t",
    )
    # Merge the HPO Terms
    df = df.drop(columns=['HPOTermParents', 'HPOTerms'])
    df_hpo = pd.read_csv("/home/sean/Data/Latest_VerifiedVariants_Annotations_GOA.tsv", sep='\t')[['VarID', 'HPOTerms', 'HPOTermParents']]
    df = df.merge(df_hpo, on='VarID')

    def has_hpo(row):
        hpo = row['HPOTerms']
        if hpo != '"HP:9999998"' and hpo != '"HP:9999999"':
            row['has_hpo'] = 1
        else:
            row['has_hpo'] = 0
        return row

    #df = df.apply(has_hpo, 1)
    #df = df.loc[df['has_hpo'] == 1]
    #df = df.drop(columns=['has_hpo'])
    #print('df with HPO:', df.shape)

    def enc_hpo(df_ann):
        df_ann["HPOTerms"] = df_ann["HPOTerms"].str.strip('"')
        df_oh = df_ann["HPOTerms"].str.get_dummies(sep=",")
        df_ann = pd.concat([df_ann, df_oh], axis=1)

        df_ann["HPOTermParents"] = df_ann["HPOTermParents"].str.strip('"')
        df_oh = df_ann["HPOTermParents"].str.get_dummies(sep=",")
        df_ann = pd.concat([df_ann, df_oh], axis=1)

        return df_ann

    if use_hpo:
        df = enc_hpo(df)
        df = df.T.groupby(level=0).sum().T
        df = df.drop(columns=['HPOTermParents', 'HPOTerms'])
        print('With HPO Terms:', df.shape)

    ## Pre-process data
    # training, testing = preprocess(df, scn_afa, kcn_afa, seed=45)
    # print("---training---")
    # print("training:", training.head(2))

    df = df.drop(columns=["HPOTerms", "HPOTermParents"], errors="ignore")

    df = df.rename(
        {"original_aa": "refAA", "substituted_aa": "altAA", "position": "pos"}
    )
    df["netChange"] = df["netChange"].map({"lof": 0, "gof": 1}).astype("int64")

    # Run this for main results 
    nested_k_fold(df, scn_afa, kcn_afa)

    # Run this for hyper parameter tuning
    #df = df.sample(frac=1).reset_index(drop=True)
    #training, testing = train_test_split(df, test_size=0.2, random_state=55, stratify=df['netChange'])
    #best_clf, best_params = fit_model_with_tuning(training, testing)


if __name__ == "__main__":
    main()
