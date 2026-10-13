"""
R3 training + full downstream analysis on a prepared cohort cache.

Runs the SAME 5-fold protocol as R2 (two-phase pretrain->DEC), with the corrected
label-aligned evaluation (RNG-neutral). Computes and saves, to
results_revision/<outdir>/:
  cv_metrics.json        per-fold silhouette/NMI/ARI (aligned) + mean/SD + full-cohort NMI/ARI
  embeddings.npy, clusters.npy
  diagnostics.json       PC1 %, effective clusters, per-sample silhouette
  cluster_validity.json  purity, Cramer's V vs cancer type AND vs batch, FDR<0.05 count
  stratified_survival.csv

Nothing is tuned. Optional --batch-correct applies leakage-safe linear batch removal per fold.
Usage: python3 r3/train.py --cohort single_archive [--batch-correct] [--smoke]
"""
import os, sys, json, pickle, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "r2")); sys.path.insert(0, ROOT)
import torch
import nbdefs as nb
from nbdefs import (config, device, SEED, HiOmicsFormer, HiOmicsLoss, HiOmicsTrainer,
                    HiOmicsConfig, MultiOmicsDataset, init_centroids)
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (silhouette_score, normalized_mutual_info_score as NMI,
                             adjusted_rand_score as ARI)
from sklearn.decomposition import PCA
from hiomicsformer_revision_addons import cluster_validity_report
from hiomicsformer_revision_survival_ext import stratified_survival_analysis


def aligned_evaluate(self, loader, labels):
    self.model.eval(); zs, ps, idx = [], [], []
    with torch.no_grad():
        for b in loader:
            ii = b["index"]
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o = self.model(b); zs.append(o["z"].cpu().numpy()); ps.append(o["q"].argmax(1).cpu().numpy())
            idx.append(ii.numpy() if hasattr(ii, "numpy") else np.asarray(ii))
    z = np.concatenate(zs); p = np.concatenate(ps); idx = np.concatenate(idx)
    order = np.argsort(idx); z, p = z[order], p[order]
    return z, p, {"silhouette": float(silhouette_score(z, p)) if len(np.unique(p)) > 1 else 0.0,
                  "nmi": float(NMI(labels, p)), "ari": float(ARI(labels, p))}
HiOmicsTrainer.evaluate = aligned_evaluate


def linear_batch_removal(Xtr, btr, ytr, Xte, bte):
    """Leakage-safe: fit per-feature (batch + type covariates) on train, remove ONLY the batch term
    from both train and test. Documented alternative to ComBat. Preserves the cancer-type covariate."""
    import numpy as _np
    def design(b, y):
        oh = _np.zeros((len(y), len(set(ytr))))
        classes = sorted(set(ytr)); ci = {c: i for i, c in enumerate(classes)}
        for i, v in enumerate(y):
            if v in ci: oh[i, ci[v]] = 1
        return _np.column_stack([_np.ones(len(y)), b.astype(float), oh])
    Dtr = design(btr, ytr)
    # solve least squares per feature: coef = (D^T D)^-1 D^T X
    coef, *_ = _np.linalg.lstsq(Dtr, Xtr, rcond=None)
    beta_batch = coef[1]                      # the batch coefficient row
    Xtr_c = Xtr - _np.outer(btr.astype(float), beta_batch)
    Xte_c = Xte - _np.outer(bte.astype(float), beta_batch)
    return Xtr_c.astype(_np.float32), Xte_c.astype(_np.float32)


def main(cohort, outdir, batch_correct, smoke, save_model=False):
    t0 = time.time()
    cache = os.path.join(HERE, f"cache_{cohort}")
    d = np.load(os.path.join(cache, "processed_data.npz")); processed = {m: d[m] for m in config.modalities}
    meta = pickle.load(open(os.path.join(cache, "meta.pkl"), "rb"))
    config.num_clusters = meta["num_clusters"]
    labels = meta["labels_encoded"]; cl = np.asarray(meta["cancer_labels"]); batch = meta["batch"]
    N = len(labels)
    out = os.path.join("results_revision", outdir); os.makedirs(out, exist_ok=True)
    print(f"[{outdir}] N={N} types={config.num_clusters} batch-correct={batch_correct}", flush=True)

    n_folds = 1 if smoke else config.n_folds
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    fold_metrics, best_val, best_state, best_fold = [], -np.inf, None, None
    feature_dims = {m: processed[m].shape[1] for m in config.modalities}

    for fi, (trv, te) in enumerate(skf.split(np.zeros(N), labels)):
        if fi >= n_folds: break
        tr, va = train_test_split(trv, test_size=0.2, stratify=labels[trv], random_state=SEED + fi)
        proc = processed
        if batch_correct:
            proc = {}
            for m in config.modalities:
                Xc = processed[m].copy()
                Xtr_c, Xte_all = linear_batch_removal(processed[m][tr], batch[tr], cl[tr],
                                                      processed[m], batch)  # apply to all rows
                Xc = Xte_all
                proc[m] = Xc
        ds = MultiOmicsDataset(proc, labels, sample_ids=meta["sample_ids"])
        mk = lambda idx: DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(idx))
        trL, vaL, teL = mk(tr), mk(va), mk(te)
        model = HiOmicsFormer(feature_dims, config).to(device)
        c1 = HiOmicsConfig(); c1.num_clusters = config.num_clusters; c1.lambda_DEC = 0.0
        c1.max_epochs = 3 if smoke else 30; c1.patience = 15
        o1 = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        HiOmicsTrainer(model, o1, HiOmicsLoss(c1), torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o1, 10, 2),
                       c1, device).train_fold(trL, vaL, labels[tr], labels[va])
        init_centroids(model, trL, config, device)
        c2 = config if not smoke else HiOmicsConfig()
        if smoke: c2.num_clusters = config.num_clusters; c2.max_epochs = 3
        o2 = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        t2 = HiOmicsTrainer(model, o2, HiOmicsLoss(c2), torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o2, 20, 2), c2, device)
        t2.train_fold(trL, vaL, labels[tr], labels[va])
        _, _, mtr = t2.evaluate(teL, labels[te])
        fold_metrics.append(mtr)
        print(f"  fold {fi+1}: Sil={mtr['silhouette']:.3f} NMI={mtr['nmi']:.3f} ARI={mtr['ari']:.3f} val={t2.best_score:.3f}", flush=True)
        if t2.best_score > best_val:
            best_val, best_fold = t2.best_score, fi
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_proc = proc

    cvsum = {mm: {"mean": float(np.mean([f[mm] for f in fold_metrics])),
                  "std": float(np.std([f[mm] for f in fold_metrics])),
                  "values": [float(f[mm]) for f in fold_metrics]} for mm in ["silhouette", "nmi", "ari"]}

    # frozen best-fold embeddings on ALL patients
    bm = HiOmicsFormer(feature_dims, config).to(device); bm.load_state_dict(best_state); bm.eval()
    fd = MultiOmicsDataset(best_proc, labels, sample_ids=meta["sample_ids"])
    embs, clus = [], []
    with torch.no_grad():
        for b in DataLoader(fd, batch_size=config.batch_size, shuffle=False):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o = bm(b); embs.append(o["z"].cpu().numpy()); clus.append(o["q"].argmax(1).cpu().numpy())
    emb = np.concatenate(embs); clusters = np.concatenate(clus)
    np.save(os.path.join(out, "embeddings.npy"), emb); np.save(os.path.join(out, "clusters.npy"), clusters)
    if save_model:
        torch.save(best_state, os.path.join(out, "best_model.pt"))

    pc = PCA(5).fit(emb).explained_variance_ratio_
    diag = {"pc1_var_pct": round(float(pc[0] * 100), 2), "pc_top5": [round(float(x * 100), 2) for x in pc],
            "effective_clusters": int((np.bincount(clusters) > 0).sum()),
            "cluster_sizes": np.bincount(clusters).tolist(),
            "full_cohort_nmi": float(NMI(cl, clusters)), "full_cohort_ari": float(ARI(cl, clusters)),
            "full_cohort_silhouette": float(silhouette_score(emb, clusters)) if len(np.unique(clusters)) > 1 else 0.0}
    json.dump({"cv_summary": cvsum, "best_fold": int(best_fold), "best_val_silhouette": float(best_val)},
              open(os.path.join(out, "cv_metrics.json"), "w"), indent=2)
    json.dump(diag, open(os.path.join(out, "diagnostics.json"), "w"), indent=2)

    # cluster validity vs cancer type AND vs batch
    import io as _io, contextlib as _cl
    with _cl.redirect_stdout(_io.StringIO()):
        cv_type = cluster_validity_report(clusters, cl)
        vbatch = None
        if len(set(batch.tolist())) > 1:
            vbatch = float(cluster_validity_report(clusters, batch)["cramers_v"])
    cval = {"purity": float(cv_type["overall_purity"]), "nmi": float(cv_type["nmi"]), "ari": float(cv_type["ari"]),
            "cramers_v_vs_cancer_type": float(cv_type["cramers_v"]),
            "cramers_v_vs_batch": vbatch,
            "n_sig_type_enrichments_fdr05": int(len(cv_type["enrichment_sig"]))}
    json.dump(cval, open(os.path.join(out, "cluster_validity.json"), "w"), indent=2)

    # stratified survival
    try:
        sd = meta["survival_data"]
        if sd is not None:
            sd = sd[~sd.index.duplicated(keep="first")]
            res = stratified_survival_analysis(embeddings=emb, cancer_labels=cl, survival_df=sd,
                                               sample_ids=meta["sample_ids"], seed=SEED)
            res.to_csv(os.path.join(out, "stratified_survival.csv"), index=False)
            import pandas as pd
            nsig = int((res.get("logrank_fdr", pd.Series(dtype=float)) < 0.05).sum())
            diag["survival_types_tested"] = int(res["logrank_p"].notna().sum())
            diag["survival_sig_fdr05"] = nsig
            json.dump(diag, open(os.path.join(out, "diagnostics.json"), "w"), indent=2)
    except Exception as e:
        print(f"  survival failed: {e}", flush=True)

    print(f"\n[{outdir}] DONE in {time.time()-t0:.0f}s")
    print(f"  CV: Sil={cvsum['silhouette']['mean']:.3f}±{cvsum['silhouette']['std']:.3f} "
          f"NMI={cvsum['nmi']['mean']:.3f} ARI={cvsum['ari']['mean']:.3f}")
    print(f"  full-cohort: NMI={diag['full_cohort_nmi']:.3f} ARI={diag['full_cohort_ari']:.3f} "
          f"PC1={diag['pc1_var_pct']}% eff_clusters={diag['effective_clusters']}")
    vb = cval['cramers_v_vs_batch']
    print(f"  cluster validity: purity={cval['purity']:.3f} V(type)={cval['cramers_v_vs_cancer_type']:.3f} "
          f"V(batch)={'n/a' if vb is None else f'{vb:.3f}'}")
    print(f"  survival: tested={diag.get('survival_types_tested')} sig(FDR<0.05)={diag.get('survival_sig_fdr05')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True); ap.add_argument("--outdir", default=None)
    ap.add_argument("--batch-correct", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    a = ap.parse_args()
    main(a.cohort, a.outdir or (a.cohort + ("_bc" if a.batch_correct else "")), a.batch_correct, a.smoke, a.save_model)
