import torch
import glob
from itertools import permutations
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import gc
from tqdm import tqdm
import sys
import os
import time
import numpy as np
from torch.nn import Softmax
import logging
from pathlib import Path
from safetensors.torch import save_file, safe_open

from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)

""" 
Run ESM models for various sequences
"""

seed = random.randrange(2**32 - 1)
random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

logname = "/home/sean/synapticafold/run_esm.log"
logging.basicConfig(
    # filename=logname,
    # filemode="a",
    format="%(asctime)s,%(msecs)d [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
    handlers=[logging.FileHandler(logname, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("--Running ESM--")


MODEL_NAME = "esm2_t36_3B_UR50D"
# MODEL_NAME = "esm2_t33_650M_UR50D"
# MODEL_NAME = "esm2_t30_150M_UR50D"
# MODEL_NAME = "esm2_t12_35M_UR50D"
# MODEL_NAME = "esm2_t6_8M_UR50D"
# MODEL_NAME = "esm1b_t33_650M_UR50S"
# MODEL_NAME = "esm2_t48_15B_UR50D"
logger.info(f"Model: {MODEL_NAME}")

last_layer = int(MODEL_NAME.split("_")[1][1:])
print("Last layer:", last_layer)

# ROOT = "/mnt/internserver1_1tb/"
ROOT = "/hdd"
# ROOT_PATH = f"{ROOT}/esm_output/{MODEL_NAME}"
ROOT_PATH = f"/home/sean/synapticafold/ESM-Function-Prediction/Data/{MODEL_NAME}/"  # _all_encoded/"
logger.info(f"Root path: {ROOT_PATH}")
Path(ROOT_PATH).mkdir(parents=False, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("Device:", device)
torch.set_float32_matmul_precision("high")
# torch.set_default_device(device)

model, alphabet = torch.hub.load("facebookresearch/esm:main", MODEL_NAME)
if device != "cpu":
    model = model.half()
model = model.eval()  # .half()  # .to(device)
model.to(device)
batch_converter = alphabet.get_batch_converter()
tokens = alphabet.all_toks
# with open(f"{ROOT}/esm_output/{MODEL_NAME}/tokens.pkl", "wb") as f:
#    pickle.dump(tokens, f)

WILDTYPE_DATA = {}


def truncate_sequence(varid: str, seq: str) -> str:
    """Truncate a sequence to 1022 elements around its mutation site

    Only needed for ESM-1b
    """
    _, aa, pos, bb = varid.split("_")
    pos = int(pos)

    new_seq = list(seq)
    assert (seq[pos - 1] == bb) or (seq[pos - 1] == aa)

    if len(new_seq) <= 1022:
        new_seq = new_seq
    elif pos < 511:
        new_seq = new_seq[:1022]
    elif pos >= len(new_seq) - 511:
        new_seq = new_seq[-1022:]
    else:
        sidx = max(0, pos - 510)
        eidx = min(len(new_seq), pos + 512)
        new_seq = new_seq[sidx:eidx]

    assert len(new_seq) <= 1022, "processed sequence too long"

    return "".join(new_seq)


def convert_to_wildtype_sequence(varid: str, seq: str) -> str:
    """Convert a mutation sequence back to its wildtype"""
    wt = list(seq)
    _, aa, pos, bb = varid.split("_")
    pos = int(pos)

    assert wt[pos - 1] == bb
    wt[pos - 1] = aa
    assert wt[pos - 1] == aa

    wt = "".join(wt)
    return wt


def convert_from_wildtype_sequence(varid: str, seq: str) -> str:
    """Generate a mutation seq given the wildtype sequence"""
    _, aa, pos, bb = varid.split("_")
    pos = int(pos)
    seq_ls = list(seq)
    assert seq_ls[pos - 1] == aa, f"Incorrect original amino acid. {varid}"
    seq_ls[pos - 1] = bb
    return "".join(seq_ls)


def run_sequence(
    varid: str, seq: str, dtype=np.float16, save_contacts: bool = False
) -> np.ndarray:
    """
    Run a variant through ESM model
    """
    # logger.info(f"> {varid}")
    data = [(varid, seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)

    t_0 = time.perf_counter()

    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=[last_layer - 1, last_layer],
            return_contacts=save_contacts,
        )

    t_1 = time.perf_counter()
    print(f"Prediction time: {(t_1 - t_0):.3f} sec")

    logits = None
    # logits = results["logits"]
    # attentions = results["attentions"]

    token_representations = results["representations"][last_layer]
    # token_rep = np.array(token_representations.cpu(), dtype=dtype)
    token_rep = token_representations
    # token_rep = token_representations.cpu().to(torch.float16)
    # logits = np.array(logits.cpu(), dtype=dtype).squeeze()

    # attentions = np.array(attentions, dtype=dtype)
    attentions = None

    contacts = None
    if save_contacts:
        contacts = results["contacts"]
        contacts = np.array(contacts.cpu(), dtype=dtype).squeeze()
        # iu = np.triu_indices(contacts.shape[0])
        # contacts = contacts[iu]

    return (token_rep, logits, contacts, attentions)


def _run_esm1(varid, seq, dtype, save_contacts):
    seq_truncated = truncate_sequence(varid, seq)
    (repr, logits, contacts, attentions) = run_sequence(
        varid, seq_truncated, dtype, save_contacts
    )
    # (repr_wt, logits_wt, contacts_wt, attentions_wt) = run_sequence(
    #    varid,
    #    truncate_sequence(varid, convert_to_wildtype_sequence(varid, seq)),
    #    dtype,
    #    save_contacts,
    # )
    # logits = logits - logits_wt
    # repr = repr - repr_wt
    # contacts = contacts - contacts_wt
    return repr, logits, contacts


def _run_esm2(varid, seq, dtype, save_contacts):
    repr, logits, contacts, attentions = run_sequence(varid, seq, dtype, save_contacts)
    return repr, logits, contacts


def encode_seq():
    tokens = [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        # ".",
        # "-",
    ]
    # genes = pd.read_csv("./merged.csv")
    genes = pd.read_csv("/home/sean/Data/wildtypes_sequences_correct.csv")
    genes = list(set(genes["gene"]))
    perms = list(permutations(tokens, 2))
    random.shuffle(perms)
    ids = {}
    for k, v in enumerate(genes):
        ids[v] = "".join(perms[k])
    return ids


def run_all2(ids: dict, save_logits=False, save_contacts=False):
    # df = pd.read_csv("./merged.csv", sep=",")
    df = pd.read_csv("./merged.csv", sep=",")
    wts = pd.read_csv("/home/sean/Data/wildtypes_sequences_correct.csv")

    # print("Loading wildtype sequences ..")
    # files = glob.glob(f"{ROOT_PATH}/*WT*")
    # for file in files:
    #    gene_name = file.split("/")[-1].split("_")[0]
    #    with safe_open(file, framework="pt", device="cuda:0") as f:
    #        e = f.get_tensor("emb")
    #        WILDTYPE_DATA[gene_name] = e

    # df = df.loc[df["VarID"] == "SCN1A_A_104_V"]

    print("Running inference ..")
    for idx, row in df.iterrows():
        varid = row["VarID"]
        gene = varid.split("_")[0]
        wt_seq = wts.loc[wts["gene"] == gene]["sequence"].item()
        seq = convert_from_wildtype_sequence(varid, wt_seq)

        gene_identity = ids[gene]
        seq = f"{gene_identity}{seq}"

        # seq = row["sequence"]
        out_path = f"{ROOT_PATH}/{varid}.safetensors"
        if os.path.exists(out_path):
            print("Skipping:", varid)
            continue
        if len(str(seq)) < 10:
            raise Exception("Short seq encountered")

        t_0 = time.perf_counter()

        if MODEL_NAME == "esm1b_t33_650M_UR50S":
            repr, logits, contacts = _run_esm1(varid, seq, np.float16, save_contacts)
        else:
            repr, logits, contacts = _run_esm2(varid, seq, np.float16, save_contacts)

        # repr_norm = repr - WILDTYPE_DATA[gene]
        # x = {"emb": repr_norm}

        x = {"emb": repr}
        save_file(x, out_path)

        t_1 = time.perf_counter()
        # logger.info(f"Inference time: ({(t_1 - t_0):.3f}sec) | {varid}")
        print(f"Inference time: ({(t_1 - t_0):.3f}sec) | {varid}")


def run_all(save_logits=False, save_contacts=False):
    outputs = {}
    # with open(
    #    "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.tsv",
    #    # "/home/sean/Arto_Annotations/Tables/MAVE_Zheng2022.csv",
    #    "r",
    # ) as f:
    with open("./merged.csv") as f:
        df = pd.read_csv(f, sep=",")

        df = df.loc[df["gene"] == "KCNQ4"]

        # df = df.loc[df["netChange"] != "neutral"]
        # print("df no neutral:", df.shape)

        for idx, row in df.iterrows():
            varid = row["VarID"]
            seq = row["sequence"]

            out_path = f"{ROOT_PATH}/{varid}.safetensors"  # _{MODEL_NAME}.npy"
            if os.path.exists(
                # out_path
                f"/home/sean/synapticafold/ESM-Function-Prediction/Data/{MODEL_NAME}/{varid}.npy"
            ):
                logger.info("Skipping, already exists")
                print(f"{varid} already exists")
                continue

            if len(str(seq)) < 10:
                logging.warn(f"Skipping, sequence too short: {seq}")
                continue

            t_0 = time.perf_counter()

            if MODEL_NAME == "esm1b_t33_650M_UR50S":
                repr, logits, contacts = _run_esm1(
                    varid, seq, np.float16, save_contacts
                )
            else:
                repr, logits, contacts = _run_esm2(
                    varid, seq, np.float16, save_contacts
                )

            # np.save(out_path, repr)
            x = {"emb": repr}
            # outputs[out_path.replace(".npy", ".safetensors")] = x
            print("out_path:", out_path)
            save_file(x, out_path)

            t_1 = time.perf_counter()
            logger.info(f"Inference time: ({(t_1 - t_0):.3f}sec) | {varid}")
            # logger.info(f"out path: {out_path}")
            # if save_logits:
            #    np.save(f"{ROOT}/esm_output/{MODEL_NAME}/{varid}_logits.npy", logits)
            #    logger.info("Saving logits")
            # if save_contacts:
            #    np.save(
            #        f"{ROOT}/esm_output/{MODEL_NAME}/{varid}_contacts.npy",
            #        contacts,
            #    )
            #    logger.info("Saving contacts")


def get_truncated_sequence():
    """Get the truncated sequence strings (ESM-1b)"""

    def prep_run(row):
        varid = row["VarID"]
        seq = row["sequence"]

        seq_truncated = truncate_sequence(varid, seq)

        with open(
            f"{ROOT}/esm_output/esm1b_t33_650M_UR50S/truncated_sequences.txt", "a"
        ) as f:
            f.write(f"{varid},{seq_truncated}\n")

    with open(
        "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.csv", "r"
    ) as f:
        df = pd.read_csv(f, sep="\t")

        df.apply(prep_run, axis=1)


def run_single_seq(varid):
    # df = pl.read_csv(
    #    "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.tsv",
    #    separator="\t",
    #    infer_schema_length=None,
    # )
    # seq = df.filter(pl.col("VarID") == varid)["sequence"].item()

    wt = pd.read_csv("/home/sean/Data/wildtypes_sequences_correct.csv")
    gene = varid.split("_")[0]
    seq_wt = wt.loc[wt["gene"] == gene]["sequence"].item()
    seq = convert_from_wildtype_sequence(varid, seq_wt)

    if MODEL_NAME == "esm1b_t33_650M_UR50S":
        repr, logits, contacts = _run_esm1(varid, seq, np.float16, save_contacts=False)
    else:
        repr, logits, contacts = _run_esm2(varid, seq, np.float16, save_contacts=False)

    # np.save(f"./{varid}.npy", repr.cpu())
    x = {"emb": repr}
    save_file(x, f"{varid}.safetensors")

    return


# The original two functions from wikipedia.
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def welford_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


def welford_finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        return float("nan")
    else:
        return (mean, variance, sampleVariance)


def welford_update(existingAggregate, newValues):
    """Batched update method - accepts 3D matrices"""
    assert newValues.ndim == 3
    (count, mean, M2) = existingAggregate
    count += newValues.shape[0]
    delta = newValues - mean
    mean += np.sum(delta / count, 0)
    delta2 = newValues - mean
    M2 += np.sum(delta * delta2, 0)
    return (count, mean, M2)


def exhaustive(gene: str):
    """Run an exhaustive search across a gene (all possible single point mutations)"""

    df = pl.read_csv(
        "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.csv",
        separator="\t",
        infer_schema_length=None,
    )

    # Get the wild type sequence for the gene
    seq = df.filter(pl.col("gene") == gene).rows(named=True)[0]
    varid = seq["VarID"]
    seq = list(seq["sequence"])
    wt = convert_to_wildtype_sequence(varid, seq)

    aa = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
    assert len(aa) == 20

    count = 0
    mean = 0
    M2 = 0
    data = []
    # for residue in range(len(seq)):
    for residue in tqdm(range(len(seq))):
        reprs = []
        for i, v in enumerate(aa):
            variant_seq = seq.copy()
            variant_seq[residue] = v
            variant_seq = "".join(variant_seq)

            out = _run_esm2(f"{residue}_{v}", variant_seq, np.float32, False)
            repr: np.ndarray = out[0].squeeze()
            reprs.append(repr)

            # update running mean and variance with welfords algorithm
            # count, mean, M2 = welford_update((count, mean, M2), repr)

            # store variant position embedding
            # +1 to account for <cls>
            emb = repr[residue + 1]
            data.append(np.array(emb))

        count, mean, M2 = welford_update((count, mean, M2), np.array(reprs))

        gc.collect()
        torch.cuda.empty_cache()

    data = np.array(data)

    w_mean, w_variance, w_sampleVariance = welford_finalize((count, mean, M2))

    agg_path = f"{ROOT}/esm_output/agg/{MODEL_NAME}/"
    Path(agg_path).mkdir(parents=False, exist_ok=True)
    with open(f"{agg_path}/{gene}_exhaustive_emb.npy", "wb") as f:
        np.save(f, data)

    with open(f"{agg_path}/{gene}_exhaustive_mean.npy", "wb") as f:
        np.save(f, w_mean)

    with open(f"{agg_path}/{gene}_exhaustive_var.npy", "wb") as f:
        np.save(f, w_sampleVariance)

    return


def run_wildtype(gene):
    df = pl.read_csv(
        "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants.csv",
        separator="\t",
        infer_schema_length=None,
    )
    # wts = pl.read_csv("/home/sean/Data/wiltype_sequences.csv")
    wts = pl.read_csv("/mnt/data2/DataFiles/wiltype_sequences.csv")

    varid = f"{gene}_WT"
    seq = wts.filter(pl.col("gene") == gene)["sequence"].item()

    if MODEL_NAME == "esm1b_t33_650M_UR50S":
        repr, logits, contacts = _run_esm1(varid, seq, np.float16, save_contacts=False)
    else:
        repr, logits, contacts = _run_esm2(varid, seq, np.float32, save_contacts=False)

    # np.save(f"./{varid}.npy", repr)
    # np.savez_compressed(f"{ROOT_PATH}/{gene}_WT.npz", a=repr)
    np.save(f"{ROOT_PATH}/{gene}_WT.npy", repr)
    return


def encode_all_variants(wt_seq: str, gene: str, df):
    seq_ls = list(wt_seq)

    vars = df.loc[df["gene"] == gene]
    """
    lof = vars.loc[df["netChange"] == "lof"]
    if lof.shape[0] > 1:
        lof = lof.sample(n=1)
        _, aa, pos, bb = lof["VarID"].item().split("_")
        pos = int(pos)
        seq_ls[pos - 1] = bb
    gof = vars.loc[df["netChange"] == "gof"]
    if gof.shape[0] > 1:
        gof = gof.sample(n=1)
        _, aa, pos, bb = gof["VarID"].item().split("_")
        pos = int(pos)
        seq_ls[pos - 1] = bb
    """

    for _, row in vars.iterrows():
        g, aa, pos, bb = row["VarID"].split("_")
        pos = int(pos)
        seq_ls[pos - 1] = bb

    return "".join(seq_ls)


def run_wildtypes(ids: dict):
    save_contacts = False
    variants_df = pd.read_csv("./merged.csv")
    df = pd.read_csv("/home/sean/Data/wildtypes_sequences_correct.csv")
    # df = pd.read_csv("/home/sean/Data/wiltype_sequences_wrong.csv")
    print(df.head())
    for idx, row in df.iterrows():
        gene = row["gene"]
        seq = row["sequence"]
        out_path = f"{ROOT_PATH}/{gene}.safetensors"

        if gene not in set(variants_df["gene"]):
            continue

        # gene_identity = ids[gene]
        # seq = f"{gene_identity}{seq}"
        # seq = list(seq)
        # seq[1] = gene_identity[0]
        # seq[2] = gene_identity[1]
        # seq = "".join(seq)
        # seq = encode_all_variants(seq, gene, variants_df)

        repr, logits, contacts = _run_esm2(gene, seq, np.float16, save_contacts)

        # WILDTYPE_DATA[gene] = repr

        x = {"emb": repr}
        # print("out_path:", out_path)
        save_file(x, out_path)
        # WILDTYPE_DATA[gene] = repr


if __name__ == "__main__":
    # run_all(save_contacts=False)
    # get_truncated_sequence()
    # run_single_seq("SCN1A_A_104_V")  # "SCN8A_A_874_T")
    # run_single_seq("KCNH2_G_584_S")
    # exhaustive("SCN1A")
    # run_wildtype("HCN2")

    ids = encode_seq()
    run_wildtypes(ids)
    # run_all2(ids, save_contacts=False)
