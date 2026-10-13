"""
PHASE 5 — Baseline sweep incl. a transformer competitor (Reviewer 1.7, 4.3).

Runs the notebook's baseline functions (Concat+KMeans, Hierarchical, Spectral,
single-modality, SNF, MOFA+ [if mofapy2], MOGONET-style) PLUS the new
SimpleMultiOmicsTransformer, all on the SAME 5-fold StratifiedKFold(SEED) splits
and the SAME metrics (silhouette / NMI / ARI). Produces:
  results_revision/baseline_comparison.csv
  figures_revision/fig4_baseline_comparison.png   (regenerated)
The HiOmicsFormer row is taken from Phase-1 cv_metrics.json.
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import torch
import nbdefs as nb
from nbdefs import (config, device, SEED, MultiOmicsDataset,
                    run_simple_baselines, run_snf_baseline, run_mofa_baseline, run_mogonet_baseline)
from hiomicsformer_revision_survival_ext import SimpleMultiOmicsTransformer
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

CACHE = os.path.join(HERE, "cache"); RES = config.results_path; FIG = config.figures_path


def log(m): print(m, flush=True)


def transformer_fold(processed, labels, feature_dims, train_idx, test_idx, full_dataset, epochs):
    tr = SimpleMultiOmicsTransformer(feature_dims, n_clusters=config.num_clusters,
                                     epochs=epochs, device=str(device), seed=SEED)
    train_loader = DataLoader(full_dataset, batch_size=config.batch_size,
                              sampler=SubsetRandomSampler(train_idx), num_workers=0)
    tr.fit_transform(train_loader)                       # train
    net = tr.net; net.eval()
    test_loader = DataLoader(full_dataset, batch_size=config.batch_size,
                             sampler=SubsetRandomSampler(test_idx), num_workers=0)
    order, embs = [], []
    with torch.no_grad():
        for batch in test_loader:
            idx = batch["index"].numpy().tolist()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            emb, _ = net(batch)
            embs.append(emb.cpu().numpy()); order.extend(idx)
    test_emb = np.concatenate(embs, axis=0)
    y = labels[np.array(order)]
    pred = KMeans(n_clusters=config.num_clusters, random_state=SEED, n_init=10).fit_predict(test_emb)
    return {"silhouette": float(silhouette_score(test_emb, pred)),
            "nmi": float(normalized_mutual_info_score(y, pred)),
            "ari": float(adjusted_rand_score(y, pred))}


def main(smoke=False):
    t0 = time.time()
    d = np.load(os.path.join(CACHE, "processed_data.npz"))
    processed = {m: d[m] for m in config.modalities}
    with open(os.path.join(CACHE, "meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    config.num_clusters = meta["num_clusters"]
    labels = meta["labels_encoded"]
    feature_dims = {m: processed[m].shape[1] for m in config.modalities}
    full_dataset = MultiOmicsDataset(processed, labels, sample_ids=meta["sample_ids"])
    N = len(labels)
    epochs = 3 if smoke else 50

    results = {}

    def add(method, m):
        results.setdefault(method, []).append(m)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(N), labels)):
        log(f"\n--- baseline fold {fold+1}/{config.n_folds} ---")
        for method, m in run_simple_baselines(processed, labels, tr_idx, te_idx, config).items():
            add(method, m)
        snf = run_snf_baseline(processed, labels, tr_idx, te_idx, config)
        if snf: add("SNF", snf)
        mofa = run_mofa_baseline(processed, labels, tr_idx, te_idx, config)
        if mofa: add("MOFA+", mofa)
        mogo = run_mogonet_baseline(processed, labels, tr_idx, te_idx, config)
        if mogo: add("MOGONET", mogo)
        tf = transformer_fold(processed, labels, feature_dims, tr_idx, te_idx, full_dataset, epochs)
        add("SimpleMultiOmicsTransformer", tf)
        log(f"  transformer: Sil={tf['silhouette']:.3f} NMI={tf['nmi']:.3f} ARI={tf['ari']:.3f}")
        if smoke:
            break

    # aggregate
    rows = []
    for method, lst in results.items():
        row = {"method": method, "n_folds": len(lst)}
        for metric in ["silhouette", "nmi", "ari"]:
            vals = [r[metric] for r in lst]
            row[f"{metric}_mean"] = float(np.mean(vals)); row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    # HiOmicsFormer row from Phase 1 — use CORRECTED (aligned) NMI/ARI, not the shuffled-
    # eval-loader values. Silhouette is order-invariant (valid); NMI/ARI are the aligned
    # full-cohort best-fold values (baselines report per-fold-test mean — noted in summary).
    try:
        cvc = json.load(open(os.path.join(RES, "cv_metrics_corrected.json")))
        rows.append({"method": "HiOmicsFormer", "n_folds": len(cvc["silhouette_per_fold"]),
                     "silhouette_mean": cvc["silhouette_mean"], "silhouette_std": cvc["silhouette_std"],
                     "nmi_mean": cvc["AUTHORITATIVE_nmi_full_cohort"], "nmi_std": 0.0,
                     "ari_mean": cvc["AUTHORITATIVE_ari_full_cohort"], "ari_std": 0.0,
                     "note": "NMI/ARI = aligned full-cohort (see cv_metrics_corrected.json)"})
    except Exception as e:
        log(f"  (HiOmicsFormer row unavailable: {e})")

    df = pd.DataFrame(rows).sort_values("silhouette_mean", ascending=False)
    df.to_csv(os.path.join(RES, "baseline_comparison.csv"), index=False)
    log("\nBaseline comparison (silhouette mean±sd):")
    for _, r in df.iterrows():
        log(f"  {r['method']:28s} Sil={r['silhouette_mean']:.3f}±{r['silhouette_std']:.3f}  "
            f"NMI={r['nmi_mean']:.3f}  ARI={r['ari_mean']:.3f}")

    _fig(df, os.path.join(FIG, "fig4_baseline_comparison.png"))
    log(f"\nPHASE 5 COMPLETE in {time.time()-t0:.0f}s")


def _fig(df, path):
    df = df.sort_values("silhouette_mean")
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.45 * len(df) + 2), sharey=True)
    for ax, metric, title in zip(axes, ["silhouette", "nmi", "ari"], ["Silhouette", "NMI", "ARI"]):
        colors = ["#d73027" if m == "HiOmicsFormer" else
                  ("#4575b4" if m == "SimpleMultiOmicsTransformer" else "#999999")
                  for m in df["method"]]
        ax.barh(y, df[f"{metric}_mean"], xerr=df[f"{metric}_std"], color=colors, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels(df["method"], fontsize=8)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel(f"{title} (mean ± SD, 5-fold)")
    fig.suptitle("Figure 4 (revised): Clustering performance vs baselines "
                 "(incl. transformer competitor)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
