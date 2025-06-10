from goatools import obo_parser
from functools import partial
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


#TARGET_FILE = "/home/sean/Arto_Annotations/Tables/MAVE_Zheng2022_annotations.csv"
#TARGET_FILE = "/home/sean/Arto_Annotations/Tables/Latest_VerifiedVariants_Annotations.tsv"
#TARGET_FILE = '../Data/pathogenic_variants_85.parquet'
#TARGET_FILE = '../inference_samples.csv'
TARGET_FILE = "./merged.csv"


def write_file(path: str, data):
    if '.parquet' in path:
        data.to_parquet(path, index=False)
    else:
        data.to_csv(path, index=False)


def read_file(path: str):
    if '.parquet' in path:
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)

def load_terms(gene_list: list[str]) -> dict:
    gene_set = set(gene_list)
    gene_to_go = {}
    with open("/home/sean/Data/goa_human_fly.gaf", "r") as f:
        for line in f:
            if line.startswith("!"):
                continue
            columns = line.strip().split("\t")
            gene_name = columns[2]  # DB_Object_Symbol
            go_id = columns[4]  # GO ID
            if gene_name in gene_set:
                if gene_name in gene_to_go:
                    gene_to_go[gene_name].add(go_id)
                else:
                    gene_to_go[gene_name] = {go_id}
    return gene_to_go


def add_to_df(gene_to_go: set):
    df = read_file(TARGET_FILE)

    def _add(row):
        # gene = row["gene"]
        gene = row["VarID"].split("_")[0]
        try:
            terms = gene_to_go[gene]
            terms = ";".join(terms)
        except KeyError:
            terms = "GO:0"
        row["GOTerms"] = terms
        return row

    df = df.apply(_add, axis=1)
    return df

def go_terms_string(gene: str) -> str:
    return ";".join(load_terms([gene])[gene])



def vectorized_add_go(varid, go_dict: dict):
    gene = varid.split('_')[0]
    try:
        terms = go_dict[gene]
        terms = sorted(terms)
        terms = ";".join(terms)
    except KeyError:
        terms = "GO:0"
    return terms


def add_go_terms(df: pd.DataFrame):
    """Add go terms to a passed dataframe"""

    gene_list = df['gene'].unique()
    gene_to_go = load_terms(gene_list)

    vectorized_add_go_with_dict = partial(vectorized_add_go, go_dict = gene_to_go)
    
    df['GOTerms'] = list(map(vectorized_add_go_with_dict, df['VarID']))
    return df

if __name__ == "__main__":

    #g = load_terms(['KCNH2'])
    #print(g)

    df = read_file(TARGET_FILE)

    def fix_gene(row):
        gene = row["VarID"].split("_")[0]
        row["gene"] = gene
        return row
    def add_gene(varid):
        return varid.split('_')[0]

    #df = df.apply(fix_gene, 1)
    df['gene'] = list(map(add_gene, df['VarID']))

    gene_list = df["gene"].unique()
    gene_to_go = load_terms(gene_list)
    print(gene_to_go)
    print('GO terms loaded')


    #df = add_to_df(gene_to_go)
    vectorized_add_go_with_dict = partial(vectorized_add_go, go_dict = gene_to_go)
    df['GOTerms'] = list(map(vectorized_add_go_with_dict, df['VarID']))
    #df['GOTerms'] = list(map(vectorized_add_go, df['VarID']))
    #df.to_csv("/home/sean/Data/Latest_VerifiedVariants_GOA.csv", index=False, sep="\t")
    dst = TARGET_FILE.split('/')[-1].split('.')[0]

    #df.to_csv(f"/home/sean/Data/{dst}_GOA.tsv", index=False, sep="\t")
    #df.to_parquet(f"../Data/pathogenic_variants_85.parquet", index=False)
    #df.to_csv(
    #    "/home/sean/Data/MAVE_Zheng2022_annotations_GOA.csv", index=False, sep="\t"
    #)
    write_file(TARGET_FILE, df)

