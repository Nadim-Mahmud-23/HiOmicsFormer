"""
E3b - Imputation / missing-modality sensitivity of the learned representation  [Reviewer 1.5].

Each variant re-runs preprocessing (r2 MultiOmicsPreprocessor logic, reproduced verbatim except for
the stated change) and trains HiOmicsFormer with the unchanged two-phase protocol on the FIRST
split of the same StratifiedKFold(SEED) used everywhere else (single split for tractability, as for
the cluster-count sweep). Variants:
  reference          4 modalities, k-NN imputation (the published pipeline)
  no_miRNA           drop the modality with the most missing values
  mean_imputation    per-feature mean instead of k-NN (k=10)
  complete_features  keep only features observed in every patient -> no imputation at all
Reported per variant: test-fold NMI/ARI/silhouette, leading-axis variance, how well the leading
axis separates the two source archives (AUC), cluster sizes, and the uncertainty coefficients of
cancer type and archive given cluster.
Output: results_revision/r4/sensitivity_<cohort>.json
"""
import argparse, os, time
import numpy as np
import pandas as pd
from common import (ROOT, OUT, config, device, SEED, log, save_json, load_raw, archive_of, association, embed)
import train as r3train  # noqa: F401  (patches HiOmicsTrainer.evaluate with the label-aligned version)
from nbdefs import HiOmicsFormer, HiOmicsLoss, HiOmicsTrainer, HiOmicsConfig, init_centroids, MultiOmicsDataset
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch

VARIANTS = {"reference": dict(mods=["mRNA", "miRNA", "Methy", "CNV"], imputer="knn", complete=False),
            "no_miRNA": dict(mods=["mRNA", "Methy", "CNV"], imputer="knn", complete=False),
            "mean_imputation": dict(mods=["mRNA", "miRNA", "Methy", "CNV"], imputer="mean", complete=False),
            "complete_features": dict(mods=["mRNA", "miRNA", "Methy", "CNV"], imputer=None, complete=True),
            "per_type_standardized": dict(mods=["mRNA", "miRNA", "Methy", "CNV"], imputer="knn", complete=False, per_type=True)}


def preprocess(omics, mods, imputer, complete, top_k=2000, per_type=False, labels=None):
    """Verbatim MultiOmicsPreprocessor.fit_transform logic, with the variant's change.
    per_type: first z-score every feature within each cancer type (i.e. within each source file),
    the convention MLOmics already applied to 11 of its 14 folders, so all files share one scale."""
    out, info = {}, {}
    for m in mods:
        df = omics[m]
        data = df.values.astype(np.float32); names = list(df.columns)
        if per_type:
            data = data.copy()
            for t in np.unique(labels):
                r = labels == t
                mu = np.nanmean(data[r], axis=0); sd = np.nanstd(data[r], axis=0, ddof=1)
                sd[~np.isfinite(sd) | (sd < 1e-8)] = np.inf          # constant within a file -> 0
                data[r] = (data[r] - mu) / sd
        if complete:
            ok = ~np.isnan(data).any(axis=0); data = data[:, ok]; names = [n for n, k in zip(names, ok) if k]
        var = np.nanvar(data, axis=0); mean = np.abs(np.nanmean(data, axis=0)) + 1e-10
        keep = np.sqrt(var) / mean >= 0.1
        data = data[:, keep]; names = [n for n, k in zip(names, keep) if k]
        k = min(top_k, data.shape[1])
        mad = np.nanmedian(np.abs(data - np.nanmedian(data, axis=0)), axis=0)
        idx = np.argsort(mad)[-k:]; data = data[:, idx]
        if np.isnan(data).any():
            data = (KNNImputer(n_neighbors=10) if imputer == "knn" else SimpleImputer(strategy="mean")).fit_transform(data)
        out[m] = StandardScaler().fit_transform(data).astype(np.float32)
        info[m] = int(data.shape[1])
    return out, info


def run_variant(name, spec, omics, labels_str, sids):
    t0 = time.time()
    config.modalities = list(spec["mods"])
    proc, dims = preprocess(omics, spec["mods"], spec["imputer"], spec["complete"],
                            per_type=spec.get("per_type", False), labels=labels_str)
    y = LabelEncoder().fit_transform(labels_str); arch = archive_of(labels_str)
    config.num_clusters = len(set(labels_str)); N = len(y)
    # model-free view of the same input: does the archive axis / type structure exist before any model?
    from sklearn.cluster import KMeans
    Xp = PCA(100, random_state=SEED).fit_transform(np.concatenate([proc[m] for m in spec["mods"]], axis=1))
    km = KMeans(n_clusters=config.num_clusters, random_state=SEED, n_init=10).fit_predict(Xp)
    inp = {"kmeans_nmi_type": float(NMI(labels_str, km)), "kmeans_ari_type": float(ARI(labels_str, km))}
    scale = np.isin(labels_str, ["GBM", "LGG", "OV"]).astype(int)   # the three files NOT z-scored within file
    if arch.min() != arch.max():
        a_in = roc_auc_score(arch, Xp[:, 0]); inp["input_pc1_auc_archive"] = float(max(a_in, 1 - a_in))
        inp["kmeans_uncertainty_archive"] = association(km, arch)["uncertainty_coefficient"]
        s_in = roc_auc_score(scale, Xp[:, 0]); inp["input_pc1_auc_scale_group"] = float(max(s_in, 1 - s_in))
    inp["kmeans_uncertainty_type"] = association(km, labels_str)["uncertainty_coefficient"]
    trv, te = next(StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED).split(np.zeros(N), y))
    tr, va = train_test_split(trv, test_size=0.2, stratify=y[trv], random_state=SEED)
    ds = MultiOmicsDataset(proc, y, sample_ids=sids)
    mk = lambda idx: DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(idx))
    trL, vaL, teL = mk(tr), mk(va), mk(te)
    model = HiOmicsFormer(dims, config).to(device)
    c1 = HiOmicsConfig(); c1.num_clusters = config.num_clusters; c1.lambda_DEC = 0.0; c1.max_epochs = 30; c1.patience = 15
    o1 = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    HiOmicsTrainer(model, o1, HiOmicsLoss(c1), torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o1, 10, 2),
                   c1, device).train_fold(trL, vaL, y[tr], y[va])
    init_centroids(model, trL, config, device)
    o2 = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    t2 = HiOmicsTrainer(model, o2, HiOmicsLoss(config), torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o2, 20, 2),
                        config, device)
    t2.train_fold(trL, vaL, y[tr], y[va])
    _, _, mt = t2.evaluate(teL, y[te])
    model.eval()
    z, q = embed(model, proc); clu = q.argmax(1)
    pca = PCA(5).fit(z); pc1 = pca.transform(z)[:, 0]
    r = {"modality_dims": dims, "input_level": inp, "test_fold": {k: float(v) for k, v in mt.items()},
         "pc1_variance_pct": float(pca.explained_variance_ratio_[0] * 100),
         "cluster_sizes": sorted([int(v) for v in np.bincount(clu) if v > 0], reverse=True),
         "full_cohort_nmi_type": float(NMI(labels_str, clu)), "full_cohort_ari_type": float(ARI(labels_str, clu)),
         "association_type": association(clu, labels_str), "minutes": round((time.time() - t0) / 60, 1)}
    if arch.min() != arch.max():
        auc = roc_auc_score(arch, pc1); r["pc1_auc_archive"] = float(max(auc, 1 - auc))
        r["association_archive"] = association(clu, arch)
        sa = roc_auc_score(scale, pc1); r["pc1_auc_scale_group"] = float(max(sa, 1 - sa))
        r["association_scale_group"] = association(clu, scale)
    np.savez_compressed(os.path.join(OUT, f"sensitivity_{COHORT}_{name}_embedding.npz"), z=z, clusters=clu, labels=labels_str)
    log(f"  {name}: test NMI={mt['nmi']:.3f} PC1={r['pc1_variance_pct']:.1f}% "
        f"AUC(archive)={r.get('pc1_auc_archive', float('nan')):.3f} sizes={r['cluster_sizes'][:4]} ({r['minutes']} min)")
    config.modalities = ["mRNA", "miRNA", "Methy", "CNV"]
    return r


COHORT = "all"


def main(cohort, variants):
    global COHORT
    COHORT = cohort
    omics, labels, sids, _ = load_raw(cohort)
    res = {}
    for v in variants:
        torch.manual_seed(SEED); np.random.seed(SEED)
        res[v] = run_variant(v, VARIANTS[v], omics, labels, sids)
        save_json(res, f"sensitivity_{cohort}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=["all", "single_archive", "all_fixed", "single_archive_fixed"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    main(a.cohort, a.variants)
