"""
E4 - Agreement with established molecular-subtype frameworks  [Reviewer 1.2].

References (independent of this study): TCGA PanCanAtlas curated subtype calls
(TCGASubtype.20170308, column Subtype_Selected: e.g. ACC CIMP classes, GI CIN/GS/MSI, GBM_LGG
methylation classes, KIRC/KIRP/LIHC/LUSC/OV/PRAD marker-paper subtypes) and the six pan-cancer
immune subtypes C1-C6 (Thorsson et al. 2018), both from the UCSC Xena PanCanAtlas hub.

Within each cancer type, patients with a known label (categories with >= 5 patients, 'NA' removed)
are partitioned by K-means with K = number of known categories on
  (a) the frozen HiOmicsFormer embedding, and
  (b) a reference: the first 50 principal components of the same patients' preprocessed input,
and agreement is scored by ARI and NMI. (b) shows how much known structure is recoverable from the
input data by a simple method; (a) whether the model's representation retains it.
Output: results_revision/r4/known_subtypes.csv, known_subtypes_summary.json
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from common import ROOT, OUT, EXT, SEED, config, log, save_json, load_cache, barcode


def references():
    sub = pd.read_csv(os.path.join(EXT, "TCGASubtype.20170308.tsv.gz"), sep="\t").set_index("sampleID")["Subtype_Selected"]
    imm = pd.read_csv(os.path.join(EXT, "Subtype_Immune_Model_Based.txt.gz"), sep="\t").set_index("sample")["Subtype_Immune_Model_Based"]
    return {"PanCanAtlas molecular subtype": sub, "Immune subtype (C1-C6)": imm}


def score(X, y, k):
    pred = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(X)
    return ARI(y, pred), NMI(y, pred)


def main():
    refs = references(); rows = []
    for model_name, cohort, emb_file in (("full-cohort model", "all", "results_revision/all_uncorrected/embeddings.npy"),
                                         ("single-archive model", "single_archive", "results_revision/single_archive/embeddings.npy")):
        processed, meta, _ = load_cache(cohort)
        emb = np.load(os.path.join(ROOT, emb_file))
        labels = np.asarray([str(x) for x in meta["cancer_labels"]])
        bcs = np.asarray([barcode(s) for s in meta["sample_ids"]])
        X_in = np.concatenate([processed[m] for m in config.modalities], axis=1)
        for ref_name, ref in refs.items():
            for t in sorted(set(labels)):
                idx = np.where(labels == t)[0]
                y = pd.Series(ref.reindex(bcs[idx]).values, index=idx).dropna().astype(str)
                y = y[~y.str.endswith(".NA") & (y != "nan")]
                vc = y.value_counts(); y = y[y.isin(vc[vc >= 5].index)]
                k = y.nunique()
                if k < 2:
                    continue
                ii = y.index.values
                a_m, n_m = score(emb[ii], y.values, k)
                Xi = PCA(min(50, len(ii) - 1), random_state=SEED).fit_transform(X_in[ii])
                a_i, n_i = score(Xi, y.values, k)
                rows.append({"model": model_name, "reference": ref_name, "cancer_type": t, "n_labelled": len(ii),
                             "k_known": k, "ARI_model": a_m, "NMI_model": n_m, "ARI_input_PCA": a_i, "NMI_input_PCA": n_i})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "known_subtypes.csv"), index=False)
    summ = {}
    for (mn, rn), g in df.groupby(["model", "reference"]):
        summ[f"{mn} | {rn}"] = {"n_types": int(len(g)), "median_ARI_model": float(g.ARI_model.median()),
                               "median_ARI_input_PCA": float(g.ARI_input_PCA.median()),
                               "median_NMI_model": float(g.NMI_model.median()),
                               "median_NMI_input_PCA": float(g.NMI_input_PCA.median()),
                               "types_model_better": int((g.ARI_model > g.ARI_input_PCA).sum())}
    log(df.round(3).to_string(index=False))
    save_json(summ, "known_subtypes_summary.json")


if __name__ == "__main__":
    main()
