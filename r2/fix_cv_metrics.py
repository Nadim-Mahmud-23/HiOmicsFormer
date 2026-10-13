"""
Produce CORRECT clustering metrics for HiOmicsFormer without retraining.

Why: the notebook's per-fold NMI/ARI (cv_metrics.json) were computed on a shuffled
eval loader with index-ordered labels -> misaligned -> NMI≈0.03/ARI≈0 (the manuscript
value Reviewer 1.2 challenged). Silhouette is order-invariant and stays valid.

This recomputes, from the SAVED best-fold model + frozen embeddings (aligned, shuffle=False):
  * per-fold silhouette (valid, from fold_predictions z/pred — order-invariant)
  * authoritative NMI/ARI vs cancer type on the FULL cohort (aligned)
  * best-fold TEST-set NMI/ARI (aligned) as a per-fold example
Writes results_revision/cv_metrics_corrected.json. Nothing is retrained; nothing faked.
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from sklearn.metrics import (silhouette_score, normalized_mutual_info_score as NMI,
                             adjusted_rand_score as ARI)

RES = "results_revision"; CACHE = os.path.join(HERE, "cache")


def main():
    emb = np.load(os.path.join(RES, "best_fold_embeddings.npy"))
    clus = np.load(os.path.join(RES, "best_fold_clusters.npy"))
    meta = pickle.load(open(os.path.join(CACHE, "meta.pkl"), "rb"))
    cl = np.asarray(meta["cancer_labels"])
    fp = pickle.load(open(os.path.join(CACHE, "fold_predictions.pkl"), "rb"))
    old = json.load(open(os.path.join(RES, "cv_metrics.json")))
    best = old["best_fold"]

    # per-fold silhouette (valid — z & pred consistently ordered)
    per_fold_sil = []
    for f, d in fp["fold_predictions"].items():
        z, pred = d["z"], d["pred"]
        per_fold_sil.append(float(silhouette_score(z, pred)) if len(np.unique(pred)) > 1 else 0.0)

    # authoritative full-cohort NMI/ARI (aligned)
    full_nmi = float(NMI(cl, clus)); full_ari = float(ARI(cl, clus))
    full_sil = float(silhouette_score(emb, clus)) if len(np.unique(clus)) > 1 else 0.0

    # best-fold test-set aligned NMI/ARI (best_fold_clusters are aligned to full order)
    ti = fp["fold_predictions"][best]["test_idx"]
    bf_nmi = float(NMI(cl[ti], clus[ti])); bf_ari = float(ARI(cl[ti], clus[ti]))

    n_eff = int((np.bincount(clus) > 0).sum())
    out = {
        "silhouette_per_fold": per_fold_sil,
        "silhouette_mean": float(np.mean(per_fold_sil)),
        "silhouette_std": float(np.std(per_fold_sil)),
        "AUTHORITATIVE_nmi_full_cohort": full_nmi,
        "AUTHORITATIVE_ari_full_cohort": full_ari,
        "full_cohort_silhouette": full_sil,
        "best_fold": best,
        "best_fold_test_nmi_aligned": bf_nmi,
        "best_fold_test_ari_aligned": bf_ari,
        "effective_clusters_used": n_eff,
        "cluster_sizes": np.bincount(clus).tolist(),
        "buggy_perfold_nmi_RETRACTED": old["cv_summary"]["nmi"],
        "buggy_perfold_ari_RETRACTED": old["cv_summary"]["ari"],
        "note": ("Per-fold NMI/ARI in cv_metrics.json are INVALID (shuffled eval loader "
                 "misaligned labels/predictions — the manuscript's NMI≈0.03 artifact). "
                 "Silhouette is order-invariant and valid. Authoritative NMI/ARI are from the "
                 "aligned full-cohort best-fold clusters. NOTE: the DEC head collapsed to "
                 f"{n_eff} effective clusters (of {len(np.unique(clus)) if False else meta['num_clusters']}), "
                 "which is why silhouette is high — temper claims of many distinct subtypes."),
    }
    json.dump(out, open(os.path.join(RES, "cv_metrics_corrected.json"), "w"), indent=2)
    print("wrote cv_metrics_corrected.json")
    print(f"  silhouette (per-fold): {out['silhouette_mean']:.3f} ± {out['silhouette_std']:.3f}")
    print(f"  AUTHORITATIVE NMI/ARI (full cohort, aligned): {full_nmi:.3f} / {full_ari:.3f}")
    print(f"  best-fold test NMI/ARI (aligned): {bf_nmi:.3f} / {bf_ari:.3f}")
    print(f"  effective clusters used: {n_eff}  sizes={np.bincount(clus).tolist()}")
    print(f"  RETRACTED buggy per-fold NMI/ARI: {out['buggy_perfold_nmi_RETRACTED']['mean']:.3f} / "
          f"{out['buggy_perfold_ari_RETRACTED']['mean']:.3f}")


if __name__ == "__main__":
    main()
