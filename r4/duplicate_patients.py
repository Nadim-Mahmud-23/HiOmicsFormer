"""
E1 - Duplicate-patient (technical-replicate) test of the archive axis  [Reviewer 1.2 / 2].

In the analysed MLOmics release the same colon patients exist twice: once in the clustering
archive (folder 'THCA', relabelled COAD and kept in the cohort) and once in the classification
archive ('GS-COAD', dropped as a duplicate). Biology is identical within each pair; only the
archive's processing differs. So any separation of the two copies is a technical (batch) effect,
independent of tissue. This breaks the archive/cancer-type confound that affects every
association statistic computed on the cohort itself.

Both copies are processed IDENTICALLY with the already-fitted full-cohort preprocessor
(transform(): fitted feature selection + scaling, fitted-mean fill for missing values), then
  (a) model-free: per-modality technical difference between copies vs. between-patient spread,
      and a patient-grouped classifier that predicts the copy's archive from its features;
  (b) model: both copies embedded by the frozen full-cohort HiOmicsFormer; position on the
      cohort's leading embedding axis (PC1) and DEC cluster assignment.
Output: results_revision/r4/duplicate_patient_test.json (+ .csv of per-patient PC1).
"""
import argparse
import numpy as np
import pandas as pd
from common import (ROOT, OUT, config, log, save_json, load_cache, load_model, embed, archive_of)
from clean_loader import CleanMLOmicsDataLoader
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import os, io, contextlib


def common_samples(omics):
    ids = None
    for df in omics.values():
        ids = set(df.index) if ids is None else ids & set(df.index)
    return sorted(ids)


def main(model_path, emb_path, clu_path):
    base = Path(config.data_path)
    loader = CleanMLOmicsDataLoader(config.data_path, config.data_variant,
                                    cache_path=os.path.join(ROOT, "r2", "cache", "clean_universe.pkl"))
    with contextlib.redirect_stdout(io.StringIO()):
        thca, _ = loader.load_single_cancer("THCA", base / "Clustering_datasets" / "THCA" / "Original",
                                            "clustering", config.modalities)
        gsco, _ = loader.load_single_cancer("COAD", base / "Classification_datasets" / "GS-COAD" / "Original",
                                            "classification", config.modalities)
    shared = sorted(set(common_samples(thca)) & set(common_samples(gsco)))
    log(f"duplicate patients present in both archives with all 4 modalities: {len(shared)}")

    # ---- raw values: exhaustive, deterministic comparison over all shared patients x features ----
    raw_cmp = {}
    for m in config.modalities:
        feats = sorted(set(thca[m].columns) & set(gsco[m].columns))
        A = thca[m].loc[shared, feats].values.astype(float)
        B = gsco[m].loc[shared, feats].values.astype(float)
        ok = ~np.isnan(A) & ~np.isnan(B)
        d = np.abs(A - B)[ok]
        raw_cmp[m] = {"shared_features": len(feats), "compared_values": int(ok.sum()),
                      "fraction_identical_(abs_diff<1e-6)": float(np.mean(d < 1e-6)),
                      "median_abs_diff": float(np.median(d)), "max_abs_diff": float(d.max()),
                      "pearson_r_all_values": float(np.corrcoef(A[ok], B[ok])[0, 1])}
    log(f"raw-value comparison of duplicate patients: {raw_cmp}")

    processed_all, meta, pre = load_cache("all")
    copies = {}
    miss = {}
    for name, om in (("clustering_copy", thca), ("classification_copy", gsco)):
        aligned = {m: om[m].loc[shared] for m in config.modalities}
        miss[name] = {m: float(aligned[m].reindex(columns=pre.get_feature_names(m)).isna().values.mean())
                      for m in config.modalities}
        copies[name] = pre.transform(aligned)
    log(f"missing fraction of the 2,000-feature panels per copy: {miss}")

    # ---- (a) model-free comparison ----------------------------------------------------------
    rng = np.random.default_rng(0)
    res_mf = {}
    n = len(shared)
    perm = rng.permutation(n)
    perm[perm == np.arange(n)] = (perm[perm == np.arange(n)] + 1) % n
    for m in config.modalities:
        A, B = copies["clustering_copy"][m], copies["classification_copy"][m]
        twin_r = np.array([np.corrcoef(A[i], B[i])[0, 1] for i in range(n)])
        other_r = np.array([np.corrcoef(A[i], A[perm[i]])[0, 1] for i in range(n)])
        twin_d = np.linalg.norm(A - B, axis=1)
        other_d = np.linalg.norm(A - A[perm], axis=1)
        X = np.vstack([A, B]); y = np.r_[np.zeros(n), np.ones(n)]; g = np.r_[np.arange(n), np.arange(n)]
        prob = np.zeros(2 * n)
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(X[tr], y[tr])
            prob[te] = clf.predict_proba(X[te])[:, 1]
        res_mf[m] = {"median_corr_same_patient_across_archives": float(np.median(twin_r)),
                     "median_corr_different_patients_same_archive": float(np.median(other_r)),
                     "median_dist_same_patient_across_archives": float(np.median(twin_d)),
                     "median_dist_different_patients_same_archive": float(np.median(other_d)),
                     "frac_pairs_twin_farther_than_random_other": float(np.mean(twin_d > other_d)),
                     "archive_auc_patient_grouped_cv": float(roc_auc_score(y, prob))}
        log(f"  {m}: twin r={res_mf[m]['median_corr_same_patient_across_archives']:.3f} "
            f"vs other r={res_mf[m]['median_corr_different_patients_same_archive']:.3f}; "
            f"archive AUC={res_mf[m]['archive_auc_patient_grouped_cv']:.3f}")

    # ---- (b) frozen full-cohort model ---------------------------------------------------------
    fd = {m: processed_all[m].shape[1] for m in config.modalities}
    model = load_model(model_path, fd, meta["num_clusters"])
    emb_cohort = np.load(emb_path)
    clu_cohort = np.load(clu_path)
    labels = np.asarray([str(x) for x in meta["cancer_labels"]])
    arch = archive_of(labels)
    pca = PCA(5).fit(emb_cohort)
    pc1_cohort = pca.transform(emb_cohort)[:, 0]
    sign = 1.0 if pc1_cohort[arch == 1].mean() > pc1_cohort[arch == 0].mean() else -1.0
    pc1_cohort = sign * pc1_cohort
    # cluster that holds the classification-archive types
    tab = pd.crosstab(clu_cohort, arch)
    cls_cluster = int(tab[1].idxmax())
    out_b = {"pc1_variance_pct": float(pca.explained_variance_ratio_[0] * 100),
             "cohort_pc1_mean_clustering_archive": float(pc1_cohort[arch == 0].mean()),
             "cohort_pc1_mean_classification_archive": float(pc1_cohort[arch == 1].mean()),
             "classification_cluster_id": cls_cluster}
    rows = []
    for name in ("clustering_copy", "classification_copy"):
        z, q = embed(model, copies[name])
        pc1 = sign * pca.transform(z)[:, 0]
        assign = q.argmax(1)
        out_b[name] = {"pc1_mean": float(pc1.mean()), "pc1_sd": float(pc1.std()),
                       "frac_assigned_to_classification_cluster": float(np.mean(assign == cls_cluster)),
                       "cluster_counts": {int(k): int(v) for k, v in zip(*np.unique(assign, return_counts=True))}}
        rows.append(pd.DataFrame({"barcode": shared, "copy": name, "pc1": pc1, "cluster": assign}))
    df = pd.concat(rows)
    df.to_csv(os.path.join(OUT, "duplicate_patient_pc1.csv"), index=False)
    p1 = df[df["copy"] == "clustering_copy"]["pc1"].values
    p2 = df[df["copy"] == "classification_copy"]["pc1"].values
    gap = out_b["cohort_pc1_mean_classification_archive"] - out_b["cohort_pc1_mean_clustering_archive"]
    out_b["paired_pc1_shift_mean"] = float(np.mean(p2 - p1))
    out_b["paired_pc1_shift_as_fraction_of_archive_gap"] = float(np.mean(p2 - p1) / gap)
    out_b["frac_pairs_classification_copy_higher_pc1"] = float(np.mean(p2 > p1))
    out_b["auc_pc1_separating_copies"] = float(roc_auc_score(np.r_[np.zeros(len(p1)), np.ones(len(p2))], np.r_[p1, p2]))
    from scipy.stats import wilcoxon
    out_b["wilcoxon_p_paired_pc1"] = float(wilcoxon(p2, p1).pvalue)
    log(f"  model: copy-separation AUC on PC1 = {out_b['auc_pc1_separating_copies']:.3f}; "
        f"classification copies in classification cluster = "
        f"{out_b['classification_copy']['frac_assigned_to_classification_cluster']:.2%} vs "
        f"clustering copies {out_b['clustering_copy']['frac_assigned_to_classification_cluster']:.2%}; "
        f"shift = {out_b['paired_pc1_shift_as_fraction_of_archive_gap']:.2f} of the archive gap")
    save_json({"n_duplicate_patients": len(shared), "raw_value_comparison": raw_cmp,
               "missing_fraction_selected_features": miss,
               "model_free": res_mf, "frozen_model": out_b}, "duplicate_patient_test.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "results_revision", "all_uncorrected", "best_model.pt"))
    ap.add_argument("--emb", default=os.path.join(ROOT, "results_revision", "all_uncorrected", "embeddings.npy"))
    ap.add_argument("--clusters", default=os.path.join(ROOT, "results_revision", "all_uncorrected", "clusters.npy"))
    a = ap.parse_args()
    main(a.model, a.emb, a.clusters)
