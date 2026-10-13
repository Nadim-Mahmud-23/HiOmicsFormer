"""
E3a - Missingness: how much of what the model saw was imputed, and does the leading embedding
axis track missingness?  [Reviewer 1.5]

For each cohort the curated pre-imputation matrices are re-loaded (identical to r3/prep.py), and
missingness is measured (i) over the whole curated feature universe and (ii) over the 2,000-feature
MAD panels actually used for training (the values the k-NN imputer filled in). Per-patient missing
fractions are then related to the frozen embeddings' leading axis (Spearman) and to the DEC clusters.
Output: results_revision/r4/missingness.json, missingness_by_type.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from common import ROOT, OUT, config, log, save_json, load_cache, load_raw, archive_of
import os


def per_patient_missing(omics, pre):
    out = {}
    for m in config.modalities:
        sel = omics[m].reindex(columns=pre.get_feature_names(m))
        out[m] = sel.isna().mean(axis=1).values
    out["all_selected"] = np.mean(np.vstack([out[m] * len(pre.get_feature_names(m)) for m in config.modalities]), axis=0) \
        * len(config.modalities) / sum(len(pre.get_feature_names(m)) for m in config.modalities)
    return out


def main():
    result, tables = {}, []
    for cohort, emb_path, clu_path in (
            ("all", "results_revision/all_uncorrected/embeddings.npy", "results_revision/all_uncorrected/clusters.npy"),
            ("single_archive", "results_revision/single_archive/embeddings.npy", "results_revision/single_archive/clusters.npy")):
        omics, labels, sids, _ = load_raw(cohort)
        _, meta, pre = load_cache(cohort)
        assert list(meta["sample_ids"]) == list(sids), "raw reload is not aligned with the cache"
        arch = archive_of(labels)
        r = {"n_patients": len(labels), "universe": {}, "selected_panel": {}}
        for m in config.modalities:
            r["universe"][m] = {"n_features": int(omics[m].shape[1]),
                                "missing_fraction": float(omics[m].isna().values.mean())}
            sel = omics[m].reindex(columns=pre.get_feature_names(m)).isna()
            r["selected_panel"][m] = {"n_features": int(sel.shape[1]),
                                      "missing_fraction": float(sel.values.mean()),
                                      "frac_patients_with_any_missing": float(sel.any(axis=1).mean())}
            if cohort == "all":
                for a, nm in ((0, "clustering_archive"), (1, "classification_archive")):
                    r["selected_panel"][m][f"missing_fraction_{nm}"] = float(sel.values[arch == a].mean())
            for t in sorted(set(labels)):
                tables.append({"cohort": cohort, "modality": m, "cancer_type": t,
                               "missing_fraction_selected": float(sel.values[labels == t].mean())})
        pm = per_patient_missing(omics, pre)
        emb = np.load(os.path.join(ROOT, emb_path)); clu = np.load(os.path.join(ROOT, clu_path))
        pcs = PCA(3).fit_transform(emb)
        r["pc_vs_missingness_spearman"] = {
            f"PC{k+1}": {m: float(spearmanr(pcs[:, k], pm[m]).correlation) for m in pm} for k in range(3)}
        if cohort == "all":
            r["missingness_auc_for_archive"] = {m: float(roc_auc_score(arch, pm[m])) for m in pm}
            r["pc1_auc_for_archive"] = float(max(roc_auc_score(arch, pcs[:, 0]), 1 - roc_auc_score(arch, pcs[:, 0])))
        sizes = pd.Series(clu).value_counts()
        if len(sizes) >= 2:
            small = int(sizes.index[-1]) if sizes.iloc[-1] >= 10 else int(sizes.index[1])
            in_small = clu == small
            r["smallest_substantive_cluster"] = {
                "cluster_id": small, "n": int(in_small.sum()),
                "composition": pd.Series(labels[in_small]).value_counts().to_dict(),
                "missing_fraction_in_cluster": {m: float(pm[m][in_small].mean()) for m in pm},
                "missing_fraction_rest": {m: float(pm[m][~in_small].mean()) for m in pm},
                "mannwhitney_p": {m: float(mannwhitneyu(pm[m][in_small], pm[m][~in_small]).pvalue)
                                  if pm[m].std() > 0 else None for m in pm}}
        result[cohort] = r
        log(f"[{cohort}] selected-panel missingness: "
            + ", ".join(f"{m}={r['selected_panel'][m]['missing_fraction']:.3f}" for m in config.modalities))
    pd.DataFrame(tables).to_csv(os.path.join(OUT, "missingness_by_type.csv"), index=False)
    save_json(result, "missingness.json")


if __name__ == "__main__":
    main()
