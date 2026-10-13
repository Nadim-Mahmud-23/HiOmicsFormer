"""
E6 + R1.1 extension.

(1) Cluster-vs-variable association, corrected  [Reviewer 1.2 / 2].
    The R2 'source archive' variable wrongly coded COAD (which came from the clustering-archive
    folder) as classification-archive. Here archive = 1 only for GBM/LGG/OV. Because archive is then a
    deterministic function of cancer type, mutual information satisfies I(C;archive) <= I(C;type);
    Cramer's V values for a 2-level and an 11-level variable are not comparable, so we report the
    uncertainty coefficient U(Y|C) = I(C;Y)/H(Y) alongside V.
(2) Silhouette in a common space. The model's silhouette is computed in its own latent space; the
    same partition is re-scored in a shared space (first 50 PCs of the preprocessed input), next to
    a K-means partition of that space.
(3) Prognostic information beyond a two-cluster split  [Reviewer 1.1]: per cancer type, five-fold
    cross-validated Cox model (L2 penalty 0.1) on the first five principal components of the frozen
    embedding, versus the same on the preprocessed input; test-fold concordance (C-index).
Output: results_revision/r4/association.json, survival_cindex.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from common import ROOT, OUT, SEED, config, log, save_json, load_cache, archive_of, association


def cv_cindex(X, time, event, n_pc=5, folds=5):
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    cs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, event):
        if event[te].sum() == 0:
            continue
        pca = PCA(n_pc, random_state=SEED).fit(X[tr])
        cols = [f"pc{i}" for i in range(n_pc)]
        dtr = pd.DataFrame(pca.transform(X[tr]), columns=cols); dtr["T"] = time[tr]; dtr["E"] = event[tr]
        cph = CoxPHFitter(penalizer=0.1).fit(dtr, "T", "E")
        risk = cph.predict_partial_hazard(pd.DataFrame(pca.transform(X[te]), columns=cols)).values
        cs.append(concordance_index(time[te], -risk, event[te]))
    return float(np.mean(cs)) if cs else float("nan"), float(np.std(cs)) if cs else float("nan")


def main():
    out = {}
    for cohort, rd in (("all", "all_uncorrected"), ("single_archive", "single_archive")):
        processed, meta, _ = load_cache(cohort)
        labels = np.asarray([str(x) for x in meta["cancer_labels"]])
        emb = np.load(os.path.join(ROOT, "results_revision", rd, "embeddings.npy"))
        clu = np.load(os.path.join(ROOT, "results_revision", rd, "clusters.npy"))
        r = {"cluster_sizes": sorted([int(v) for v in np.bincount(clu) if v > 0], reverse=True),
             "vs_cancer_type": association(clu, labels)}
        arch = archive_of(labels)
        if arch.min() != arch.max():
            r["vs_archive"] = association(clu, arch)
            r["crosstab_cluster_by_archive"] = pd.crosstab(clu, arch).to_dict()
        X = np.concatenate([processed[m] for m in config.modalities], axis=1)
        Xp = PCA(50, random_state=SEED).fit_transform(X)
        k = len(set(labels))
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Xp)
        r["silhouette_model_partition_in_model_space"] = float(silhouette_score(emb, clu))
        r["silhouette_model_partition_in_input_pca_space"] = float(silhouette_score(Xp, clu))
        r["silhouette_kmeans_partition_in_input_pca_space"] = float(silhouette_score(Xp, km))
        r["silhouette_cancer_type_labels_in_input_pca_space"] = float(silhouette_score(Xp, labels))
        out[cohort] = r
        log(f"[{cohort}] U(type|C)={r['vs_cancer_type']['uncertainty_coefficient']:.3f} "
            + (f"U(archive|C)={r['vs_archive']['uncertainty_coefficient']:.3f} " if 'vs_archive' in r else "")
            + f"sil model-space={r['silhouette_model_partition_in_model_space']:.3f} "
              f"input-space={r['silhouette_model_partition_in_input_pca_space']:.3f}")
    save_json(out, "association.json")

    rows = []
    for cohort, rd, model_name in (("all", "all_uncorrected", "full-cohort model"),
                                   ("single_archive", "single_archive", "single-archive model")):
        processed, meta, _ = load_cache(cohort)
        labels = np.asarray([str(x) for x in meta["cancer_labels"]])
        emb = np.load(os.path.join(ROOT, "results_revision", rd, "embeddings.npy"))
        X = np.concatenate([processed[m] for m in config.modalities], axis=1)
        sd = meta["survival_data"]; sd = sd[~sd.index.duplicated(keep="first")]
        ids = [s.split("_", 1)[1] if "_" in s else s for s in meta["sample_ids"]]
        for t in sorted(set(labels)):
            idx = np.array([i for i in np.where(labels == t)[0] if ids[i] in sd.index])
            if len(idx) < 30:
                continue
            T = sd.loc[[ids[i] for i in idx], "survival_times"].astype(float).values
            E = sd.loc[[ids[i] for i in idx], "event_observed"].astype(int).values
            ok = T > 0; idx, T, E = idx[ok], T[ok], E[ok]
            if E.sum() < 10:
                rows.append({"model": model_name, "cancer_type": t, "n": len(idx), "events": int(E.sum()),
                             "cindex_model": np.nan, "cindex_input": np.nan, "note": "<10 events"})
                continue
            cm, sm = cv_cindex(emb[idx], T, E); ci, si = cv_cindex(X[idx], T, E)
            rows.append({"model": model_name, "cancer_type": t, "n": len(idx), "events": int(E.sum()),
                         "cindex_model": cm, "cindex_model_sd": sm, "cindex_input": ci, "cindex_input_sd": si, "note": ""})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "survival_cindex.csv"), index=False)
    log(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
