import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)


def load_output(run: str, model: str = "esm"):
    root = "/home/sean/synapticafold/ESM-Function-Prediction/"
    if model == "esm":
        try:
            df = pd.read_parquet(f"{root}/Log/{run}/output.parquet")
            return df, "parquet"
        except FileNotFoundError:
            df = pd.read_csv(f"{root}/Log/{run}/output.csv", sep="\t")
            return df, "csv"
        raise FileNotFoundError("No output file found")
    elif model == "svm":
        df = pd.read_csv(f"{root}/Experiments/svm/Log/{run}/output.csv")
        return df, "csv"


@dataclass
class Fold:
    """
    Each fold holds predictions for all epochs within it.
    We can index into the fold to get the predictions for a specific epoch
    or also get the predictions across a set of folds
    """

    fold: int
    preds: np.ndarray  # [n_epochs, samples]
    labels: np.ndarray  # [n_epochs, samples]
    varids: List[List]

    def __getitem__(self, idx: int):
        # Index into a specific epoch within the fold
        return self.labels[idx, :], self.preds[idx, :]

    def __len__(self):
        return self.preds.shape[0]

    def __repr__(self) -> str:
        return f"fold:{self.fold} - {self.preds.shape}"

    def compute_scores(self):
        self.mcc = []
        self.roc_auc = []
        for epoch in range(self.preds.shape[0]):
            l = self.labels[epoch]
            p = self.preds[epoch]
            p = (p >= 0.5).astype(int)
            mcc = matthews_corrcoef(l, p)
            self.mcc.append(mcc)
            roc_auc = roc_auc_score(l, p)
            self.roc_auc.append(roc_auc)

    def cross_entropy(self, predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
        return ce

    def get_best_epoch(self) -> int:
        self.cross_entropy_loss = []
        for epoch in range(self.preds.shape[0]):
            c = self.cross_entropy(self.preds[epoch], self.labels[epoch])
            self.cross_entropy_loss.append(c)
        self.best_epoch = np.argmin(self.cross_entropy_loss)
        return self.best_epoch


def create_dataset_svm(df: pd.DataFrame) -> List[Fold]:
    out = []
    for fold in df["fold"].unique():
        sub = df[df["fold"] == fold]
        pred = np.array(list(sub["pred"])).reshape(1, -1)
        label = np.array(list(sub["label"])).reshape(1, -1)
        varid = list(sub["VarID"])

        f = Fold(fold, pred, label, varid)
        out.append(f)
    return out


def create_dataset(df: pd.DataFrame) -> List[Fold]:
    out = []
    for fold in df["fold"].unique():
        sub = df[(df["mode"] == "eval") & (df["fold"] == fold)]

        varids = []
        preds = []
        labels = []
        for epoch in sub["epoch"].unique():
            sub_e = sub[sub["epoch"] == epoch]
            pred = np.array(list(sub_e["pred"]))
            label = np.array(list(sub_e["label"]))

            pred = np.exp(pred) / np.sum(np.exp(pred), 1)[:, None]
            # pred = pred[:, 1]

            # label = np.argmax(label, 1)

            preds.append(pred)
            labels.append(label)

            varid = sub_e["VarID"].to_list()
            varids.append(varid)

        preds = np.array(preds)
        labels = np.array(labels)
        f = Fold(fold, preds, labels, varids)
        out.append(f)
    return out


def compute_roc2(
    folds: List[Fold], ax: plt.axes, name: str, epochs: List[int] = [19], **kwargs
):
    """Compute the Receiver Operating Characteristic for the deep learning model outputs"""
    assert len(epochs) == len(folds)

    ps = []
    ls = []
    tprs = []
    aucs = []
    thresholds = []
    thresholds_best = []
    for ii, fold in enumerate(folds):
        epoch = epochs[ii]
        p = fold.preds[epoch]
        l = fold.labels[epoch]

        p = p[:, 1]
        l = np.argmax(l, 1)

        # p = np.concatenate(p)
        # l = np.concatenate(l)

        ps.append(p)
        ls.append(l)

        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, threshold = roc_curve(l, p)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        thresholds.append(threshold)
        thresholds_best.append(np.argmax(tpr - fpr))
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, 0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    linestyle = kwargs.get("linestyle", "-")
    include_label = kwargs.get("include_label", True)

    label = f"{name} (AUC={mean_auc:.3f}+-{std_auc:.3f})"
    plot_kwargs = kwargs.get("plot_kwargs", {})
    if not include_label:
        label = None
    if ax:
        ax.plot(mean_fpr, mean_tpr, label=label, **plot_kwargs)
    return aucs, thresholds, thresholds_best


def compute_roc(
    folds: List[Fold], ax: plt.axes, name: str, epochs: List[int] = [19], **kwargs
):
    """Compute the Receiver Operating Characteristic for the SVM model outputs"""
    ps = []
    ls = []
    tprs = []
    aucs = []
    for fold in folds:
        p = fold.preds[epochs]
        l = fold.labels[epochs]

        p = np.concatenate(p)
        l = np.concatenate(l)

        ps.append(p)
        ls.append(l)

        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, threshold = roc_curve(l, p)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, 0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    linestyle = kwargs.get("linestyle", "-")

    if ax:
        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{name} (AUC={mean_auc:.3f}+-{std_auc:.3f}",
            linestyle=linestyle,
            **kwargs,
        )
    return aucs


def interpolate_zeros(vector):
    # Find the indices of zeros, excluding the first and last elements
    zero_indices = np.where(
        (vector == 0)
        & (np.arange(len(vector)) != 0)
        & (np.arange(len(vector)) != len(vector) - 1)
    )[0]

    # If there are no interior zeros, return the original vector
    if len(zero_indices) == 0:
        return vector

    # Create a copy of the vector to modify
    result = vector.copy()

    # Iterate through the zero indices
    for idx in zero_indices:
        # Find the previous and next non-zero values
        prev_value = result[idx - 1]
        next_value = result[idx + 1 :][result[idx + 1 :] != 0][0]
        next_idx = np.where(result[idx + 1 :] != 0)[0][0] + idx + 1

        # Interpolate
        interpolated_value = prev_value + (next_value - prev_value) * (
            idx - (idx - 1)
        ) / (next_idx - (idx - 1))

        # Replace the zero with the interpolated value
        result[idx] = interpolated_value

    return result


def compute_pr(
    folds: List[Fold], ax: plt.axes, name: str, epochs: List[int] = [19], **kwargs
):
    """Computes the Precision Recall curve"""
    ls = []
    ps = []
    for fold in folds:
        p = fold.preds[epochs]
        l = fold.labels[epochs]
        p = np.concatenate(p)
        l = np.concatenate(l)

        ls.append(l)
        ps.append(p)

    ps = np.concatenate(ps)
    ls = np.concatenate(ls)

    precision, recall, thresholds = precision_recall_curve(ls, ps)
    score = average_precision_score(ls, ps)
    print(f"{name}: pr auc:", score)

    # To make the plot nicer, I interpolate incase there are any precision drops to 0
    precision = interpolate_zeros(precision)

    ax.plot(recall, precision, label=f"{name} (AUC={score:.3f})", **kwargs)


if __name__ == "__main__":
    pass
