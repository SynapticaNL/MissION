import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datetime
from typing import List, Dict
import shutil
import glob
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import random
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
)
import socket
import argparse
import logging
from sklearn.preprocessing import StandardScaler
import sklearn.utils as sku
from safetensors import safe_open
from pathlib import Path
import utils as U
from Preprocessing import preprocessing as P
from Preprocessing import add_goa
from Models import models, pre_train_model, focalloss

# Init argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM training")
    parser.add_argument("-l", "--log", type=str, help="Log folder name", required=True)
    parser.add_argument(
        "--ignore_ann",
        action="store_false",
        help="If passed, then do NOT use the annotations in the model",
    )
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument(
        "--cut_size",
        type=int,
        help="Number of columns to take around mutation column from esm embeddings",
        default=200,
    )
    parser.add_argument(
        "-mr",
        "--mave_ratio",
        type=int,
        default=0,
        help="Ratio of mave to add to train set",
    )
    args = parser.parse_args()
    print("Logging to:", args.log)

    print("Cut size:", args.cut_size)

    Path(f"./Log/{args.log}/").mkdir(parents=True, exist_ok=True)

    # Logging
    logname = f"/home/sean/synapticafold/ESM-Function-Prediction/Log/{args.log}/log"
    logging.basicConfig(
        # filename=logname,
        # filemode="a",
        format="%(asctime)s,%(msecs)d [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.FileHandler(logname, mode="a"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("---Training Model---")
    logger.info(f"{datetime.datetime.now().isoformat()}")
    logger.info(f"cmd line args: {args}")

    logger.info(f"cut size: {args.cut_size}")

    def set_seeds(seed=None):
        if not seed:
            seed = 42  # 44
            seed = 9778741310
            seed = 7357005351
            seed = random.randrange(2**32 - 1)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        logger.info(f"seed: {seed}")
        return rng, seed

    rng, seed = set_seeds()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.set_float32_matmul_precision("high")
    # torch.set_default_device(device)
    # torch._dynamo.config.cache_size_limit = 100

    # Load datasets
    ESM_MODEL_NAME = "esm2_t36_3B_UR50D"
    # ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
    # ESM_MODEL_NAME = 'esm2_t30_150M_UR50D'
    # ESM_MODEL_NAME = "esm2_t12_35M_UR50D"
    # ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
    # ESM_MODEL_NAME = "ProteinBert"
    DATA_ROOT = "/home/sean/synapticafold/ESM-Function-Prediction/Data/"
    logger.info(f"MODEL: {ESM_MODEL_NAME}")

    # ROOT = "/mnt/internserver1_data/"
    if socket.gethostname() == "internserver1":
        DATA_ROOT = "/mnt/data/"

    BATCH_SIZE = 8
    DIMS = {
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t6_8M_UR50D": 320,
        "ProteinBert": 1562,
    }
    FEATURE_DIM = DIMS[ESM_MODEL_NAME]
    # FEATURE_DIM = 1024 # prott5

    USE_ANN = args.ignore_ann
    logger.info(f"Using ann: {USE_ANN}")

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"FEATURE_DIM: {FEATURE_DIM}")


def slice_with_padding(matrix, r, window_size=10):
    """Cut out a window around position r
    Returns a w*2 sized window
    """
    rows, cols = matrix.shape

    # Calculate start and end indices
    start = max(0, r - window_size)
    end = min(rows, r + window_size + 1)

    # Calculate padding
    pad_top = max(0, window_size - r)
    pad_bottom = max(0, (r + window_size + 1) - rows)

    # Slice the matrix
    sliced = matrix[start:end, :]

    # Create a mask
    mask = torch.zeros((2 * window_size + 1), device=matrix.device)
    # Set 1s in the padding regions of the mask
    if pad_top > 0:
        mask[:pad_top] = 1
    if pad_bottom > 0:
        mask[-pad_bottom:] = 1

    # Pad if necessary
    # if pad_top > 0 or pad_bottom > 0:
    #    sliced = np.pad(
    #        sliced, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0
    #    )
    if pad_top > 0 or pad_bottom > 0:
        sliced = torch.nn.functional.pad(
            sliced, (0, 0, pad_top, pad_bottom), mode="constant", value=0
        )

    return sliced, mask


def gene_lengths() -> Dict[str, int]:
    sizes = {}
    df = pd.read_csv("/home/sean/Data/wildtypes_sequences_correct.csv")
    for _, row in df.iterrows():
        gene = row["gene"]
        seq = row["sequence"]
        # Add 2 to account for BOS/EOS
        sizes[gene] = len(seq) + 2
    return sizes


def find_mod_pos(pos: int, gene_length: int):
    if gene_length <= 1022:
        return pos
    if pos < 511:
        return pos
    if pos >= gene_length - 511:
        return 1024 - (gene_length - pos) - 1
    return 512


def slice_no_center(emb: torch.tensor, P: int, W: int) -> torch.tensor:
    """Cut of a W sized window, without going outside the sequence"""
    # assert emb.ndim == 2
    num_cols = emb.shape[0]
    start = max(0, P - W // 2)
    end = min(num_cols, P + W // 2)  # +1 because the end in slicing is exclusive

    # If P is near the start, adjust to ensure W*2 width
    if P - W < 0:
        end = min(W, num_cols)

    # If P is near the end, adjust to ensure W*2 width
    if P + W >= num_cols:
        start = max(0, num_cols - W)

    ne = emb[start:end, :]
    if ne.shape[0] < W:
        ne = torch.nn.functional.pad(
            ne, (0, 0, (W - ne.shape[0]), 0), mode="constant", value=0
        )
    return ne


def collate_cutout(x: list[tuple]):
    variant = [i["variant"] for i in x]
    gene = [i["gene"] for i in x]
    label = [i["label"] for i in x]
    emb = [i["emb"] for i in x]
    annotations = [i["annotation_data"] for i in x]
    pos = [i["pos"] for i in x]
    msa = [i["msa"] for i in x]
    contacts = [i["contacts"] for i in x]
    go_terms = [i["go_terms"] for i in x]

    # Process the embeddings to be equal size for stacking
    CUT_OUT_SIZE = args.cut_size  # 100
    new_emb = []
    masks = []
    for i in range(len(emb)):
        ne, mask = slice_with_padding(emb[i][0], pos[i], CUT_OUT_SIZE)  # [None]
        ne = ne[None]
        mask = mask[None]
        # ne = slice_no_center(emb[i][0], pos[i], CUT_OUT_SIZE)
        new_emb.append(ne)
        masks.append(mask)

    ## Cut out around MSA
    ## Compute the location of the mutation col
    # gene_sizes = gene_lengths()
    # new_msa = []
    # for i in range(len(msa)):
    #    m = msa[i][0]  # [1024, 768]
    #    M = pos[i].item()
    #    L = gene_sizes[gene[i]]
    #    clipped_mod_pos = find_mod_pos(M, L)

    #    ne, _ = slice_with_padding(m, clipped_mod_pos, CUT_OUT_SIZE)#[None]
    #    new_msa.append(ne[None])

    # new_emb = emb

    go_terms = torch.tensor([0], device="cpu")
    # go_terms = torch.cat(go_terms)

    # masks = torch.tensor([0], device='cpu')
    msa = torch.tensor([0], device="cpu")
    # msa = torch.stack(new_msa).squeeze(1)
    new_emb = torch.stack(new_emb).squeeze(1)  # -> [bs, 2560]
    masks = torch.stack(masks).squeeze(1)
    annotations = torch.stack(annotations)[:, 0, :]  # -> [bs, ann_dim]
    label = torch.stack(label)[:, 0, :]  # -> [bs, 2]
    return (
        variant,
        gene,
        label,
        new_emb,
        annotations,
        masks,
        msa,
        go_terms,
    )  # pos, msa, contacts


def collate_comb(x: list[tuple]):
    """Batch collate function

    Combines the mutation column from the missense variant with the wildtype
    """
    variant = [i["variant"] for i in x]
    gene = [i["gene"] for i in x]
    label = [i["label"] for i in x]
    emb = [i["emb"] for i in x]
    annotations = [i["annotation_data"] for i in x]
    pos = [i["pos"] for i in x]
    msa = [i["msa"] for i in x]
    contacts = [i["contacts"] for i in x]

    # Seperate out the emb
    # emb_wt = emb[1, :, :] # [res, 2560]
    # emb = emb[0, :, :] # [res, 2560]

    new_emb = []
    for i in range(len(emb)):
        ne = emb[i][0]  # [res, 2560]
        ne_wt = emb[i][1]  # [res, 2560]

        # ne_mean = torch.mean(ne, 0)
        # wt_mean = torch.mean(ne_wt, 0)

        ne = ne[int(pos[i]), :]  # [2560]
        wt = ne_wt[int(pos[i]), :]  # [2560]
        ne_norm = ne - wt

        # comb = torch.cat((ne_norm, wt, ne_mean, wt_mean), 0)
        # comb = torch.stack((ne_norm, wt, ne_mean, wt_mean), dim=0)
        comb = torch.stack((ne_norm, wt), dim=0)
        new_emb.append(comb)

    new_emb = torch.stack(new_emb)  # .squeeze(1)  # -> [bs, 2560]
    # masks = torch.stack(masks).squeeze(1)
    masks = torch.Tensor([])
    annotations = torch.stack(annotations)[:, 0, :]  # -> [bs, ann_dim]
    label = torch.stack(label)[:, 0, :]  # -> [bs, 2]
    return variant, gene, label, new_emb, annotations, masks  # pos, msa, contacts


class GeneDatasetLazy(Dataset):
    """Dataset for loading data during training

    Takes a pandas dataframe of variants with the necessary data and id columns
    """

    def __init__(
        self,
        variants: pd.DataFrame,
        use_esm: bool = True,
        use_msa: bool = False,
        use_contacts: bool = False,
    ):
        self.variants: List[Dict] = variants[["VarID", "gene", "netChange"]].to_dict(
            "records"
        )
        self.annotations = self.df_to_dict(
            variants.drop(
                columns=["gene", "netChange", "sequence", "GOTerms"], errors="ignore"
            ),
            "VarID",
        )
        if "GOTerms" in set(variants.columns):
            self.go_terms = self.df_to_dict(variants[["VarID", "GOTerms"]], "VarID")
            self.go_max_len = 0
            for k, v in self.go_terms.items():
                if len(v[0]) > self.go_max_len:
                    self.go_max_len = len(v[0])
            print("Longest go:", self.go_max_len)
        else:
            self.go_terms = False

        self.wt = {}
        self.use_esm = use_esm
        self.use_msa = use_msa
        self.use_contacts = use_contacts

    @staticmethod
    def df_to_dict(df, key_column: str) -> Dict:
        data_dict = {}
        for idx, row in df.iterrows():
            key = row[key_column]
            vals = [row[col] for col in df.columns if col != key_column]
            data_dict[key] = vals
        return data_dict

    def __len__(self):
        return len(self.variants)

    def load_esm_emb(self, varid):
        with safe_open(
            f"./Data/{ESM_MODEL_NAME}/{varid}.safetensors", framework="pt", device="cpu"
        ) as f:
            emb = f.get_tensor("emb")

        """
        with safe_open(f"./Data/{ESM_MODEL_NAME}/{gene}_WT.safetensors", framework='pt', device='cpu') as f:
            embwt = f.get_tensor('emb')  # [1, res, 2560]
        emb = emb - embwt
        """

        return emb

    def load_msa_emb(self, varid):
        """Return the MSA-Transformer embeddings"""
        with open(
            f"/home/sean/Projects/ESMTransformer/output/{varid}_repr.npz",
            "rb",
        ) as f:
            emb = np.load(f)
            # emb_wt = emb["wt"]
            emb = emb["a"]
            # emb -= emb_wt
            # emb = torch.tensor(emb, device="cpu")
            # emb = emb[None]
            return torch.tensor(emb, device="cpu")[None]

    def load_esm_contacts(self, varid):
        """Return the ESM2 contact predictions"""
        _, _, pos, _ = varid.split("_")
        pos = int(pos)
        with open(
            f"/home/sean/synapticafold/ESM-Function-Prediction/Data/{ESM_MODEL_NAME}_contacts/{varid}_contacts.npy",
            "rb",
        ) as f:
            contacts = np.load(f)
            contacts = contacts[int(pos) - 1, :]
            contacts = torch.tensor(contacts, device="cpu")
            return contacts[None]

    def pad_embedding_family(self, emb: torch.Tensor, gene: str) -> torch.Tensor:
        """Pad the input to its gene family's max size"""
        family_size_max = {"SC": 2016, "CA": 2506, "KC": 1236, "SH": 655, "HC": 889}

        assert emb.ndim == 3, "Assume emb has shape (batch, n_res, n_emb)"
        emb_shape = emb.shape[1]
        diff = family_size_max[gene[:2]] - emb_shape
        if diff == 0:
            return emb
        return F.pad(emb, (0, 0, 0, diff))

    def __getitem__(self, idx: int):
        """Returns a single sample from the variants dataframe for training"""
        x = self.variants[idx]
        gene = x["gene"]
        varid = x["VarID"]
        annotation_data = torch.tensor(
            self.annotations[varid],
            device="cpu",
        )
        go_terms = []
        if self.go_terms:
            go_terms = torch.tensor(
                self.go_terms[varid], device="cpu", dtype=torch.long
            )
            go_terms = F.pad(
                go_terms, (0, self.go_max_len - go_terms.shape[1]), "constant", 0
            )

        pos = torch.tensor(int(varid.split("_")[2]), dtype=torch.int32, device="cpu")

        # Loading embeddings
        emb = None
        if self.use_esm:
            emb = self.load_esm_emb(varid)  # [1, seq_len+2, FEATURE_DIM]

        # Load MSA embeddings
        msa = None
        if self.use_msa:
            msa = self.load_msa_emb(varid)  # [<=1024, 768]

        # Load ESM2 contact preds
        contacts = None
        if self.use_contacts:
            contacts = self.load_esm_contacts(varid)  # [seq_len, seq_len]

        label = torch.tensor(x["netChange"], device="cpu")
        label = F.one_hot(label, num_classes=2) * 1.0

        # Add batch dim
        annotation_data = annotation_data[None]
        label = label[None]

        # return x, gene, label, emb, annotation_data, pos, msa, contacts
        return {
            "variant": x,
            "gene": gene,
            "label": label,
            "emb": emb,
            "annotation_data": annotation_data,
            "pos": pos,
            "msa": msa,
            "contacts": contacts,
            "go_terms": go_terms,
        }


def accuracy(y_pred: torch.Tensor, y: torch.Tensor):
    """Computes accuracy of model predictions during training"""
    y_pred = torch.argmax(y_pred, dim=-1)
    y = torch.argmax(y, dim=-1)
    count = torch.sum(y_pred == y, dim=0)
    return count / y_pred.shape[0]


def parameters_to_1d(params):
    return torch.cat([x.view(-1) for x in params])


def l2_loss(layer_parameters, alpha: float):
    """Return the L2 loss for a given layer
    layer_parameters: model.layer_name.parameters()
    """
    flat = torch.cat([x.view(-1) for x in layer_parameters])
    return alpha * torch.norm(flat, 1)


def train_batch(
    model,
    train_dataset,
    test_dataset,
    optimizer,
    criterion,
    n_epochs,
    scheduler=None,
    **kwargs,
):
    """Primary model training function

    Runs forward and backpropagation to train a model
    """
    fold = kwargs.get("fold", 0)
    archive = kwargs.get("archive", None)
    dataset_mave = kwargs.get("dataset_mave", None)

    for epoch in range(n_epochs):
        model.train()
        t_0 = time.perf_counter()
        loss_hist_epoch = []
        acc_hist_epoch = []
        for batch_idx, batch in enumerate(train_dataset):
            variant, gene, label, emb, ann, mask, msa, go_terms = batch
            # variant, gene, label, emb, ann = batch
            # mask = None
            emb = emb.to(device, non_blocking=True)
            ann = ann.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            msa = msa.to(device, non_blocking=True)
            # go_terms = go_terms.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # with torch.autocast(device_type='cpu', dtype=torch.float16):
                y_pred = model(
                    emb, ann, gene=gene, mask=mask, msa=msa, go_terms=go_terms
                )

            acc = accuracy(y_pred, label)
            loss = criterion(y_pred, label)

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # loss += l2_loss(model.l_emb[0].parameters(), 0.001)
            loss_ = loss.detach().item()
            acc_ = acc.detach().item()
            loss_hist_epoch.append(loss_)
            acc_hist_epoch.append(acc_)

            loss = 0

        model.eval()
        eval_preds = []
        eval_labels = []
        eval_acc_hist = []
        eval_variants = []
        test_loss = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataset):
                variant, gene, label, emb, ann, mask, msa, go_terms = batch
                # variant, gene, label, emb, ann = batch
                eval_variants.extend(variant)
                emb = emb.to(device, non_blocking=True)
                ann = ann.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                msa = msa.to(device, non_blocking=True)
                # go_terms = go_terms.to(device, non_blocking=True)
                # mask = None
                eval_labels.append(label)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # y_pred = model(emb, ann, gene=gene)
                    y_pred = model(
                        emb, ann, gene=gene, mask=mask, msa=msa, go_terms=go_terms
                    )

                eval_preds.append(y_pred.detach())

                acc = accuracy(y_pred, label)
                loss = criterion(y_pred, label)
                eval_acc_hist.append(acc.detach().item())
                test_loss.append(loss.detach().item())

            if dataset_mave:
                mave_eval_preds = []
                mave_eval_labels = []
                mave_eval_variants = []
                mave_acc = []
                mave_loss = []
                for batch_idx, batch in enumerate(dataset_mave):
                    variant, gene, label, emb, ann = batch
                    mave_eval_variants.extend(variant)
                    emb = emb.to(device, non_blocking=True)
                    ann = ann.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    mave_eval_labels.append(label)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        y_pred = model(emb, ann, gene=gene)

                    mave_eval_preds.append(y_pred.detach())
                    acc = accuracy(y_pred, label)
                    loss = criterion(y_pred, label)
                    mave_acc.append(acc.detach().item())
                    mave_loss.append(loss.detach().item())

        t_1 = time.perf_counter()

        eval_preds = torch.cat(eval_preds, 0).cpu()
        eval_labels = torch.cat(eval_labels, 0).cpu()

        for i in range(len(eval_preds)):
            archive.log(
                {
                    "VarID": eval_variants[i]["VarID"],
                    "gene": eval_variants[i]["gene"],
                    "mode": "eval",
                    "fold": fold,
                    "epoch": epoch,
                    "pred": eval_preds[i].tolist(),
                    "label": eval_labels[i].tolist(),
                }
            )

        if dataset_mave:
            mave_eval_preds = torch.cat(mave_eval_preds, 0).cpu()
            mave_eval_labels = torch.cat(mave_eval_labels, 0).cpu()
            for i in range(len(mave_eval_preds)):
                archive.log(
                    {
                        "VarID": mave_eval_variants[i]["VarID"],
                        "gene": mave_eval_variants[i]["gene"],
                        "mode": "eval",
                        "fold": fold,
                        "epoch": epoch,
                        "pred": mave_eval_preds[i].tolist(),
                        "label": mave_eval_labels[i].tolist(),
                    }
                )

        def eval_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
            f1 = U.f1_score(y_true, y_pred, average=None)
            mcc = U.phi_coefficient(y_true, y_pred)
            return f1, mcc

        f1, mcc = eval_metrics(eval_labels, eval_preds)

        if dataset_mave:
            f1_mave, mcc_mave = eval_metrics(mave_eval_labels, mave_eval_preds)

        s = f"Epoch: {epoch} | {(t_1 - t_0):.2f}s | train_loss: {np.mean(loss_hist_epoch):.3f} | test_loss: {np.mean(test_loss):.3f} | train_acc: {np.mean(acc_hist_epoch):.2f} \
| acc: {np.mean(eval_acc_hist):.2f} | mcc: {mcc:.3f} | f1: {f1} | grad: {norm:.3f}"

        if dataset_mave:
            s += f" | mave mcc: {mcc_mave:.3f} | mave f1: {f1_mave}"

        print(s)

    return archive


@U.timeit
def normalise_df(
    df: pd.DataFrame,
    exclude: List = ["VarID", "gene", "netChange", "sequence"],
    scaler=None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalise numerical cols in df

    Will temporarily remove columns specified in the exclude list,
    then try to normalise the remaining ones, before returning the excluded
    columns
    """
    # ids = df[["VarID", "netChange", "gene"]]
    id_cols = df[exclude]
    df = df.drop(columns=exclude)
    cols = df.columns
    x = df.to_numpy()
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    x = imp.fit_transform(x)
    if not scaler:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        x_scaled = scaler.transform(x)
    df_scaled = pd.DataFrame(x_scaled, columns=cols)
    df_scaled = pd.concat((id_cols, df_scaled), axis=1)
    return df_scaled, scaler  # , cols


class Archive:
    """Class used for logging inference results to disk"""

    def __init__(self, filepath: str, verbose: bool = False):
        self.filepath = filepath
        self.output = []
        self.verbose = verbose

    def log(self, row: dict):
        self.output.append(row)

    def flush(self):
        out = pd.DataFrame.from_dict(self.output)
        if ".csv" in self.filepath:
            out.to_csv(self.filepath, index=False)
        elif ".parquet" in self.filepath:
            out.to_parquet(self.filepath, index=False)
        else:
            logger.error(
                f"Archive received incorrect filetype {self.filepath} - logging as csv"
            )
            out.to_csv(self.filepath, index=False)
        print(f"Archive ({out.shape}) flushed to: {self.filepath}")


def init_dataset(fold, use_esm=True, use_msa=False, use_contacts=False, shuffle=False):
    """Returns a pytorch dataloader created from a dataset"""
    temp_dataset = GeneDatasetLazy(
        fold,
        use_esm=use_esm,
        use_msa=use_msa,
        use_contacts=use_contacts,
    )

    dataloader = DataLoader(
        temp_dataset,
        batch_size=8,
        shuffle=shuffle,
        num_workers=5,
        # num_workers = 1,
        prefetch_factor=BATCH_SIZE * 8,
        # prefetch_factor=1,
        collate_fn=collate_cutout,
        # collate_fn=collate_comb,
        # collate_fn=collate_pad,
        pin_memory=True,
    )
    return dataloader


def run(
    train_fold: pd.DataFrame,
    test_fold: pd.DataFrame,
    sizes,
    hist: List = [],
    fold: int = 0,
    archive=None,
    **kwargs,
):
    """Trains a model for a single cross-validation fold of data

    Handles creating the torch dataset, instantiating the model and any necessary helpers (optimizer, loss fn)
    """

    mave_fold = kwargs.get("mave_fold", None)
    # Class weights
    y = train_fold["netChange"].to_list()
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    print("class weight:", cw)
    class_weights = torch.tensor(cw, device=device)

    use_esm = True
    use_msa = False
    use_contacts = False
    dataloader_train = init_dataset(
        train_fold,
        use_esm=use_esm,
        use_msa=use_msa,
        use_contacts=use_contacts,
        shuffle=True,
    )
    dataloader_test = init_dataset(
        test_fold,
        use_esm=use_esm,
        use_msa=use_msa,
        use_contacts=use_contacts,
        shuffle=False,
    )

    dataloader_mave = None
    if mave_fold is not None:
        dataloader_mave = init_dataset(
            mave_fold,
            use_esm=use_esm,
            use_msa=use_msa,
            use_contacts=use_contacts,
            shuffle=False,
        )

    # Takes a small section around the mutation col - same for all genes
    model = models.SimpleMLP4(
        sizes,
        input_dim=FEATURE_DIM,
        emb_dim=int(FEATURE_DIM * 0.1),  # 256,
        ann_dim=train_fold.shape[1] - 3,
        # ann_dim = train_fold.shape[1] - 4, # kept GOTerms for embedding
        ann_emb_dim=256,
        out_dim=2,
        pad_size=args.cut_size,
        squeeze_dim=32,
        use_ann=USE_ANN,
    ).to(device)

    epochs = args.epochs
    logger.info(f"Epochs: {epochs}")

    print("params:", U.count_parameters(model))
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = focalloss.sigmoid_focal_loss(reduction="sum")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=0.0001
    )  # 0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=1e-5, T_max=epochs * (len(dataloader_train) / BATCH_SIZE)
    )
    print("len(dataloader_train):", len(dataloader_train))
    # scheduler = None

    archive = train_batch(
        model,
        dataloader_train,
        dataloader_test,
        optimizer,
        criterion,
        n_epochs=epochs,
        scheduler=scheduler,
        fold=fold,
        archive=archive,
        dataset_mave=dataloader_mave,
    )
    archive.flush()
    return None, archive


def filter_uncommon(df: pd.DataFrame, cut_off: int = 10):
    """Filter out genes with a low number of variants"""
    ls = []
    ug = df["gene"].unique()
    for gene in ug:
        sub = df.loc[df["gene"] == gene]
        if sub.shape[0] < cut_off:
            ls.append(gene)
    df = df.loc[~df["gene"].isin(ls)]
    return df


def kfold_cv():  # genes: pd.DataFrame, sizes: dict):
    """Run cross-validation on the data"""

    set_seeds(65412)

    genes = P.prepare_dataset(
        "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.tsv",
        # "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants_Annotations.tsv",
        "/home/sean/Data/Latest_VerifiedVariants_Annotations_GOA.tsv",
        f"./Data/{ESM_MODEL_NAME}/",
        basic_features := True,
        expression_levels := False,
        include_hpo := False,
        include_go := True,
        keep_only_hpo := False,
    )
    print(genes.shape)

    include_mtl = False
    if include_mtl:
        genes = add_mtl_data(genes)
    print(genes.shape)
    print(genes.head())

    sizes = P.get_gene_sizes(genes)
    print("genes:", genes.shape)

    # Shuffle genes list
    genes = sku.shuffle(genes, random_state=seed)
    genes = genes.reset_index(drop=True)

    groups = np.array(genes["gene"])
    n_groups = len(set(groups))
    print("n_groups:", n_groups)

    # kf = KFold(n_splits=10, random_state=None)
    # kf = StratifiedKFold(n_splits=10)
    kf = GroupKFold(n_splits=n_groups)
    hist = []
    archive = Archive(f"./Log/{args.log}/output.parquet", verbose=True)
    for fold_idx, (train_index, test_index) in enumerate(
        kf.split(genes, y=genes["netChange"].to_list(), groups=groups)
    ):
        # print(f"Fold: {i}", train_index, test_index)
        print(f"--== Fold: {fold_idx} ==--")
        print("Test gene:", groups[test_index][0])

        train_fold = genes.iloc[train_index]
        test_fold = genes.iloc[test_index]
        train_fold = train_fold.reset_index(drop=True)
        test_fold = test_fold.reset_index(drop=True)

        print("labels train:", np.bincount(train_fold["netChange"]))
        print("labels test:", np.bincount(test_fold["netChange"]))

        # Normalise
        if basic_features or include_go or include_hpo or expression_levels:
            train_fold, scaler = normalise_df(
                train_fold,
                exclude=["VarID", "gene", "netChange", "sequence"],
                scaler=None,
            )
            test_fold, _ = normalise_df(
                test_fold,
                exclude=["VarID", "gene", "netChange", "sequence"],
                scaler=scaler,
            )

        train_fold = train_fold.drop(columns=["sequence"])
        test_fold = test_fold.drop(columns=["sequence"])

        print("train_fold:", train_fold.shape)
        print("test_fold:", test_fold.shape)

        hist, archive = run(
            train_fold, test_fold, sizes, hist=hist, fold=fold_idx, archive=archive
        )


def add_predicted_hpo(df):
    import ast

    with open("./Log/hpo_predictions/predicted_hpo_terms.json", "r") as f:
        predicted = ast.literal_eval(f.read())

    def add(row):
        if row["HPOTerms"] == '"HP:9999998"':
            row["HPOTerms"] = f'"{predicted[row["VarID"]]}"'
        return row

    df = df.apply(add, 1)
    return df


def add_mtl_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the processed bosselman matrix into the genes dataframe"""
    # mtl = pd.read_csv("./Experiments/svm/processed_feature_matrix_MTL_2024-08-05.csv")
    mtl = pd.read_csv("./Data/processed_feature_matrix_MTL_2024-08-05.csv")
    df = pd.merge(df, mtl, on="VarID")
    return df


def merge_mave(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the mave data into our gene set"""
    mave = pd.read_csv(
        "/home/sean/Arto_Annotations/Tables/MAVE_Zheng2022.csv", sep="\t"
    )
    mave = mave[["VarID", "gene", "netChange", "sequence"]]
    mave_ann = pd.read_csv(
        # "/home/sean/Arto_Annotations/Tables/MAVE_Zheng2022_annotations.csv", sep="\t"
        "/home/sean/Data/MAVE_Zheng2022_annotations_GOA.csv",
        sep="\t",
    )
    mave = pd.merge(mave, mave_ann, on="VarID")
    mave["train"] = 2
    mave = mave.loc[mave["netChange"] != "neutral"]

    # Remove any KCNQ4 already in the gene list
    df = df.loc[df["gene"] != "KCNQ4"]

    c = df.columns
    c2 = mave.columns
    c_shared = [i for i in c if i in c2]
    df = df[c_shared]
    mave = mave[c_shared]

    df = pd.concat([df, mave])
    df.loc[df["gene"] == "KCNQ4", "train"] = 0
    print("Genes + Mave:", df.shape)
    df = df.reset_index(drop=True)
    return df


def sample_from_mave(df, percentage=None):
    if percentage > 0 and percentage < 1:
        sub_lof = (
            df.loc[(df["gene"] == "KCNQ4") & (df["netChange"] == "lof")]
            .sample(frac=percentage)
            .index
        )
        sub_gof = (
            df.loc[(df["gene"] == "KCNQ4") & (df["netChange"] == "gof")]
            .sample(frac=percentage)
            .index
        )
    elif percentage >= 1:
        percentage = int(percentage)
        sub_lof = (
            df.loc[(df["gene"] == "KCNQ4") & (df["netChange"] == "lof")]
            .sample(percentage)
            .index
        )
        sub_gof = (
            df.loc[(df["gene"] == "KCNQ4") & (df["netChange"] == "gof")]
            .sample(percentage)
            .index
        )

    df.loc[sub_lof, "train"] = 1
    df.loc[sub_gof, "train"] = 1

    return df


def sample_from_mave_2(df, goi: str, n=10, percentage=None):
    """
    Adds n samples of the gene of interest (goi) to test-set (2)
    Sets percentage% samples from each class of the gene to train-set (1)
    Remainder is added to test-set (0)
    """
    df.loc[df["gene"] == goi, "train"] = 0
    print(f"Subsampling =={goi}== for testing and increasing training splits")
    print(f"Test samples: {n}. Training percentage: {percentage}")
    temp_rng = np.random.default_rng(404)  # old seed: 512
    df = df.reset_index(drop=True)

    # Test set (fixed size, always the same)
    sub_lof = (
        df.loc[(df["gene"] == goi) & (df["netChange"] == "lof")]
        .sample(n=n, random_state=temp_rng)
        .index
    )
    sub_gof = (
        df.loc[(df["gene"] == goi) & (df["netChange"] == "gof")]
        .sample(n=n, random_state=temp_rng)
        .index
    )

    # Set all genes=GOI to 0(test)
    df.loc[(df["gene"] == goi), "train"] = 0

    # Set the above as the test set
    df.loc[sub_lof, "train"] = 2
    df.loc[sub_gof, "train"] = 2

    # Train set (sample some percentage)
    sub_lof = (
        df.loc[(df["gene"] == goi) & (df["netChange"] == "lof") & (df["train"] != 2)]
        # .sample(frac=percentage)#n=percentage)
        .sample(n=percentage)
        .index
    )
    sub_gof = (
        df.loc[(df["gene"] == goi) & (df["netChange"] == "gof") & (df["train"] != 2)]
        # .sample(frac=percentage)#n=percentage)
        .sample(n=percentage)
        .index
    )

    df.loc[sub_lof, "train"] = 1
    df.loc[sub_gof, "train"] = 1
    return df


def balance_mave(df):
    """Balance the MAVE samples to have some specified ratio of variants in each class"""
    df.loc[df["gene"] == "KCNQ4", "train"] = 0
    df = df.reset_index(drop=True)
    sub_lof = (
        df.loc[(df["gene"] == "KCNQ4") & (df["netChange"] == "lof")].sample(200).index
    )
    sub_gof = (
        df.loc[(df["gene"] == "KCNQ4") & (df["netChange"] == "gof")].sample(100).index
    )

    df.loc[sub_lof, "train"] = 2
    df.loc[sub_gof, "train"] = 2
    df = df.reset_index(drop=True)

    sub_lof = (
        df.loc[
            (df["gene"] == "KCNQ4") & (df["netChange"] == "lof") & (df["train"] == 0)
        ]
        .sample(1)
        .index
    )
    sub_gof = (
        df.loc[
            (df["gene"] == "KCNQ4") & (df["netChange"] == "gof") & (df["train"] == 0)
        ]
        .sample(1)
        .index
    )
    df.loc[sub_lof, "train"] = 1
    df.loc[sub_gof, "train"] = 1
    df = df.reset_index(drop=True)

    return df


def fixed_test_set():
    """Train the model on some data while holding out a fixed set of variants for testing"""
    archive = Archive(f"./Log/{args.log}/output.parquet", verbose=True)

    for fold_idx, state in enumerate(
        [404, 505]  # , 606, 707, 808, 909, 1010, 1111, 1212, 1313]
    ):
        df = pd.read_csv("./merged.csv")

        basic_featurse = True
        include_hpo = False
        include_go = True
        include_mtl = False
        expression_levels = False
        # genes = add_predicted_hpo(genes)
        df = P.prepare_annotations(
            df,
            basic_features=basic_featurse,
            expression_levels=expression_levels,
            include_hpo=include_hpo,
            include_go=include_go,
        )
        sizes = P.get_gene_sizes(df)

        n = 5
        goi = "KCNA2"
        df["train"] = 1
        # Hold out 10 SCN5A variants
        test_lof = (
            df.loc[(df["gene"] == goi) & (df["netChange"] == "lof")]
            .sample(n=n, random_state=state)
            .index
        )
        test_gof = (
            df.loc[(df["gene"] == goi) & (df["netChange"] == "gof")]
            .sample(n=n, random_state=state)
            .index
        )
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace({"netChange": {"lof": 0, "gof": 1}})

        df.loc[test_lof, "train"] = 0
        df.loc[test_gof, "train"] = 0

        # genes = sku.shuffle(genes, random_state=seed)
        # genes = genes.reset_index(drop=True)

        # train_fold = df.loc[(df["train"] == 1) & (df["gene"].str.contains("KCN"))]
        train_fold = df.loc[df["train"] == 1]
        test_fold = df.loc[df["train"] == 0]

        print("train head:\n", train_fold[["VarID", "netChange", "train"]].head())
        print("train tail:\n", train_fold[["VarID", "netChange", "train"]].tail())

        train_fold = train_fold.drop(
            columns=["train", "sequence", "substituted_aa", "position", "original_aa"],
            errors="ignore",
        ).reset_index(drop=True)
        test_fold = test_fold.drop(
            columns=["train", "sequence", "substituted_aa", "position", "original_aa"],
            errors="ignore",
        ).reset_index(drop=True)

        print("labels train:", np.bincount(train_fold["netChange"]))
        print("labels test:", np.bincount(test_fold["netChange"]))

        if basic_featurse or include_hpo or include_go:
            print("Normalising train_fold and test_fold")
            train_fold, scaler = normalise_df(
                train_fold, exclude=["VarID", "gene", "netChange"], scaler=None
            )
            test_fold, _ = normalise_df(
                test_fold, exclude=["VarID", "gene", "netChange"], scaler=scaler
            )
            # mave_fold, _ = normalise_df(
            #    mave_fold, exclude=["VarID", "gene", "netChange"], scaler=scaler
            # )
        mave_fold = None

        hist = []
        hist, archive = run(
            train_fold,
            test_fold,
            sizes,
            hist=hist,
            fold=fold_idx,
            archive=archive,
            mave_fold=mave_fold,
        )


def custom_folds():
    """Runs cross-validation on pre-split folds of data

    See Preproecssing/split_data to see how folds are generated
    """
    folds = sorted(glob.glob("./Data/data_splits/repeated_k_fold/fold_*"))

    archive = Archive(f"./Log/{args.log}/output.parquet", verbose=True)

    hist = []
    for fold in folds:
        p = Path(fold).stem
        print(f"--== {p} ==--")
        _, fold_idx = p.split("_")

        genes = pd.read_csv(fold, sep="\t")
        sizes = P.get_gene_sizes(genes)

        basic_featurse = True
        include_hpo = False
        include_go = True
        include_mtl = False
        expression_levels = False
        # genes = add_predicted_hpo(genes)
        genes = P.prepare_annotations(
            genes,
            basic_features=basic_featurse,
            expression_levels=expression_levels,
            include_hpo=include_hpo,
            include_go=include_go,
        )
        print(genes.head())
        print("GOTerms included:", "GOTerms" in genes.columns)

        if include_mtl:
            genes = add_mtl_data(genes)

        print("Include BF:", basic_featurse)
        print("Include HPO:", include_hpo)
        print("Include GO:", include_go)
        print("Include expression_levels:", expression_levels)

        with pd.option_context("future.no_silent_downcasting", True):
            genes = genes.replace({"netChange": {"lof": 0, "gof": 1}})

        genes = genes.reset_index(drop=True)
        print("final df:", genes.shape)
        print(genes.head())

        mave_fold = None
        train_fold = genes.loc[genes["train"] == 1]
        test_fold = genes.loc[genes["train"] == 0]
        # mave_fold = genes.loc[genes["train"] == 2]
        print("labels train:", np.bincount(train_fold["netChange"]))
        print("labels test:", np.bincount(test_fold["netChange"]))
        # print("labels mave:", np.bincount(mave_fold["netChange"]))

        train_fold = train_fold.drop(
            columns=["train", "sequence", "substituted_aa", "position", "original_aa"],
            errors="ignore",
        )
        test_fold = test_fold.drop(
            columns=["train", "sequence", "substituted_aa", "position", "original_aa"],
            errors="ignore",
        )
        # mave_fold = mave_fold.drop(columns=["train", "sequence"])

        train_fold = train_fold.reset_index(drop=True)
        test_fold = test_fold.reset_index(drop=True)

        # mave_fold = mave_fold.reset_index(drop=True)
        if basic_featurse or include_hpo or include_go:
            print("Normalising train_fold and test_fold")
            train_fold, scaler = normalise_df(
                train_fold, exclude=["VarID", "gene", "netChange"], scaler=None
            )
            test_fold, _ = normalise_df(
                test_fold, exclude=["VarID", "gene", "netChange"], scaler=scaler
            )
            if mave_fold:
                mave_fold, _ = normalise_df(
                    mave_fold, exclude=["VarID", "gene", "netChange"], scaler=scaler
                )

        hist, archive = run(
            train_fold,
            test_fold,
            sizes,
            hist=hist,
            fold=fold_idx,
            archive=archive,
            mave_fold=mave_fold,
        )


def sample_gene(genes: pd.DataFrame, gene: str, amount: int):
    """Sample a subset of variants from a specified gene"""
    sub_lof = (
        genes.loc[(genes["gene"] == gene) & (genes["netChange"] == "lof")]
        .sample(amount)
        .index
    )
    sub_gof = (
        genes.loc[(genes["gene"] == gene) & (genes["netChange"] == "gof")]
        .sample(amount)
        .index
    )

    genes.loc[sub_lof, "train"] = 1
    genes.loc[sub_gof, "train"] = 1

    genes.reset_index(drop=True, inplace=True)
    return genes


def sub_sample_genes(df):
    """Sub-samples the majority class to only have as many samples as the minority class"""
    genes = list(set(df["gene"]))
    counts = df.groupby("gene")["netChange"].value_counts().unstack(fill_value=0)

    samples = []
    for gene in genes:
        v = counts.loc[gene].tolist()
        g = v[0]
        l = v[1]
        min_ = min(g, l)

        if min_ == 0:
            continue

        sub = df.loc[df["gene"] == gene]
        gof_samples = sub.loc[sub["netChange"] == "gof"].sample(
            n=min_, ignore_index=True
        )
        lof_samples = sub.loc[sub["netChange"] == "lof"].sample(
            n=min_, ignore_index=True
        )
        samples.append(gof_samples)
        samples.append(lof_samples)

    genes = pd.concat(samples)
    print("Genes after subsampling minority class:", genes.shape)
    return genes


def save_files():
    """Store current training files"""
    # Save a copy of the files being run
    shutil.copy2(models.__file__, f"./Log/{args.log}/models.py")
    shutil.copy2(P.__file__, f"./Log/{args.log}/preprocessing.py")
    shutil.copy2(__file__, f"./Log/{args.log}/train3.py")


def filter_AM(genes, annotations, filter_val: float = 0.3):
    """Filter out variants with low AlphaMissense score"""
    ann = annotations.to_pandas()
    new = []
    for idx, row in genes.iterrows():
        if row["gene"].upper() == "SH":
            new.append(row)
            continue
        varid = row["VarID"]
        AM = ann.loc[ann["VarID"] == varid]["AM"].item()

        if AM > filter_val:
            new.append(row)
    genes = pd.DataFrame(new)
    genes = genes.drop(genes.columns[0], axis=1)
    genes.to_csv("./genes_am_filtered.csv", index=False)
    return genes


if __name__ == "__main__":
    save_files()

    # kfold_cv()  # genes, sizes)
    # fixed_test_set()
    custom_folds()

    plt.show()
