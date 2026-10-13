"""
PHASE 1 — 5-fold CV training + frozen best-fold embeddings (Reviewer 4.2 continuity).

Reuses the notebook's HiOmicsFormer / HiOmicsTrainer / loss (via nbdefs) and the exact
two-phase (pretrain -> DEC fine-tune) protocol from notebook Cell 9. Adds:
  * best fold selected by VALIDATION silhouette (trainer.best_score)
  * frozen embeddings + cluster assignments for ALL patients from the best-fold model,
    row-aligned to cancer_labels / sample_ids
Saves:
  results_revision/best_fold_embeddings.npy   (N, latent_dim)
  results_revision/best_fold_clusters.npy     (N,)
  results_revision/cv_metrics.json            (per-fold + mean/SD)
  r2/cache/fold_predictions.pkl, r2/cache/best_model.pt

Determinism: keeps the notebook's fixed seeds. Usage:
  python3 r2/phase1_train.py            # full 5-fold run
  python3 r2/phase1_train.py --smoke    # 1 fold, tiny epochs (pipeline check only)
"""
import os, sys, json, time, pickle, argparse, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)             # nbdefs
sys.path.insert(0, ROOT)             # revision add-on modules

import torch
import nbdefs as nb
from nbdefs import (config, device, SEED, HiOmicsFormer, HiOmicsLoss, HiOmicsTrainer,
                    HiOmicsConfig, MultiOmicsDataset, init_centroids)
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

CACHE = os.path.join(HERE, "cache")


def log(m): print(m, flush=True)


# ---------------------------------------------------------------------------
# BUGFIX (Reviewer 1.2): the notebook's HiOmicsTrainer.evaluate() iterates a
# SHUFFLED eval loader (SubsetRandomSampler) but scores NMI/ARI against labels
# kept in index order -> labels and predictions are MISALIGNED, driving the
# reported NMI≈0.03 / ARI≈0. Silhouette is order-invariant so it stays valid.
# This override collects each sample's dataset index and re-sorts predictions to
# the (ascending) label order before scoring. It touches only numpy metric math
# (no torch RNG draw), so a re-run reproduces the identical trained model while
# reporting CORRECT NMI/ARI.
# ---------------------------------------------------------------------------
def _aligned_evaluate(self, loader, labels):
    self.model.eval()
    zs, preds, idxs = [], [], []
    with torch.no_grad():
        for batch in loader:
            idx = batch["index"]
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = self.model(batch)
            zs.append(out["z"].cpu().numpy())
            preds.append(out["q"].argmax(dim=1).cpu().numpy())
            idxs.append(idx.numpy() if hasattr(idx, "numpy") else np.asarray(idx))
    z = np.concatenate(zs); pred = np.concatenate(preds); idx = np.concatenate(idxs)
    order = np.argsort(idx)            # ascending dataset index == sorted subset == labels order
    z, pred = z[order], pred[order]
    metrics = {
        "silhouette": float(silhouette_score(z, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
    }
    return z, pred, metrics


HiOmicsTrainer.evaluate = _aligned_evaluate


def load_cache():
    d = np.load(os.path.join(CACHE, "processed_data.npz"))
    processed = {m: d[m] for m in config.modalities}
    with open(os.path.join(CACHE, "meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    config.num_clusters = meta["num_clusters"]
    return processed, meta


def main(smoke=False):
    t0 = time.time()
    processed, meta = load_cache()
    labels_encoded = meta["labels_encoded"]
    cancer_labels = meta["cancer_labels"]
    sample_ids = meta["sample_ids"]
    survival_data = meta["survival_data"]

    full_dataset = MultiOmicsDataset(processed, labels_encoded,
                                     survival_data=survival_data, sample_ids=sample_ids)
    feature_dims = full_dataset.get_feature_dims()
    N = len(full_dataset)
    log(f"Loaded cache: N={N}, feature_dims={feature_dims}, K={config.num_clusters}")

    n_folds = 1 if smoke else config.n_folds
    max_ep = 3 if smoke else None  # None => use config defaults

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    fold_results, fold_predictions, fold_histories = [], {}, {}
    best_val_sil, best_state, best_fold = -np.inf, None, None

    for fold_idx, (train_val_idx, test_idx) in enumerate(
            skf.split(np.zeros(N), labels_encoded)):
        if fold_idx >= n_folds:
            break
        log("\n" + "-" * 70); log(f"FOLD {fold_idx+1}/{n_folds}  (train+val={len(train_val_idx)}, test={len(test_idx)})")

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2,
            stratify=labels_encoded[train_val_idx], random_state=SEED + fold_idx)

        def mk(idx, shuffle_sampler=True):
            return DataLoader(full_dataset, batch_size=config.batch_size,
                              sampler=SubsetRandomSampler(idx), num_workers=0)
        train_loader, val_loader, test_loader = mk(train_idx), mk(val_idx), mk(test_idx)

        model_fold = HiOmicsFormer(feature_dims, config).to(device)

        # --- Phase 1: pretrain (recon only) ---
        cfg_p1 = HiOmicsConfig(); cfg_p1.num_clusters = config.num_clusters
        cfg_p1.lambda_DEC = 0.0; cfg_p1.max_epochs = 3 if smoke else 30; cfg_p1.patience = 15
        opt1 = torch.optim.AdamW(model_fold.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt1, T_0=10, T_mult=2)
        tr1 = HiOmicsTrainer(model_fold, opt1, HiOmicsLoss(cfg_p1), sch1, cfg_p1, device)
        h1 = tr1.train_fold(train_loader, val_loader, labels_encoded[train_idx], labels_encoded[val_idx])

        init_centroids(model_fold, train_loader, config, device)

        # --- Phase 2: DEC fine-tune ---
        cfg_p2 = config
        if smoke:
            cfg_p2 = HiOmicsConfig(); cfg_p2.num_clusters = config.num_clusters; cfg_p2.max_epochs = 3
        opt2 = torch.optim.AdamW(model_fold.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=20, T_mult=2)
        tr2 = HiOmicsTrainer(model_fold, opt2, HiOmicsLoss(cfg_p2), sch2, cfg_p2, device)
        h2 = tr2.train_fold(train_loader, val_loader, labels_encoded[train_idx], labels_encoded[val_idx])

        z_test, pred_test, test_metrics = tr2.evaluate(test_loader, labels_encoded[test_idx])
        fold_results.append(test_metrics)
        fold_predictions[fold_idx] = {"test_idx": test_idx, "pred": pred_test, "z": z_test}
        fold_histories[fold_idx] = {"p1": h1, "p2": h2}
        val_sil = float(tr2.best_score)
        log(f"  fold {fold_idx+1}: test Sil={test_metrics['silhouette']:.4f} "
            f"NMI={test_metrics['nmi']:.4f} ARI={test_metrics['ari']:.4f} | val_sil(best)={val_sil:.4f}")

        if val_sil > best_val_sil:
            best_val_sil, best_fold = val_sil, fold_idx
            best_state = {k: v.cpu().clone() for k, v in model_fold.state_dict().items()}

        # incremental checkpoint
        with open(os.path.join(CACHE, "fold_predictions.pkl"), "wb") as fh:
            pickle.dump({"fold_predictions": fold_predictions, "fold_results": fold_results,
                         "fold_histories": fold_histories}, fh)

    # ---- CV summary ----
    cv_summary = {}
    for metric in ["silhouette", "nmi", "ari"]:
        vals = [r[metric] for r in fold_results]
        cv_summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                              "values": [float(v) for v in vals]}
    log("\nCV summary: " + json.dumps({k: f"{v['mean']:.3f}±{v['std']:.3f}" for k, v in cv_summary.items()}))
    log(f"Best fold by val silhouette: fold {best_fold+1} (val_sil={best_val_sil:.4f})")

    # ======================================================================
    # Frozen best-fold embeddings for ALL patients (row-aligned to labels/ids)
    # ======================================================================
    best_model = HiOmicsFormer(feature_dims, config).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    embs, clus = [], []
    with torch.no_grad():
        for batch in full_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = best_model(batch)
            embs.append(out["z"].cpu().numpy())            # mu at eval (encode)
            clus.append(out["q"].argmax(dim=1).cpu().numpy())
    best_embeddings = np.concatenate(embs, axis=0)
    best_cluster_labels = np.concatenate(clus, axis=0)
    assert best_embeddings.shape[0] == N == len(cancer_labels) == len(sample_ids), "alignment broken"
    assert np.isfinite(best_embeddings).all(), "NaN/Inf in embeddings"

    np.save(os.path.join(config.results_path, "best_fold_embeddings.npy"), best_embeddings)
    np.save(os.path.join(config.results_path, "best_fold_clusters.npy"), best_cluster_labels)
    torch.save(best_state, os.path.join(CACHE, "best_model.pt"))
    with open(os.path.join(config.results_path, "cv_metrics.json"), "w") as fh:
        json.dump({"cv_summary": cv_summary, "best_fold": int(best_fold),
                   "best_val_silhouette": best_val_sil,
                   "best_fold_full_silhouette": float(
                       silhouette_score(best_embeddings, best_cluster_labels)
                       if len(np.unique(best_cluster_labels)) > 1 else 0.0),
                   "smoke": smoke}, fh, indent=2)

    log(f"\nSaved best_fold_embeddings.npy {best_embeddings.shape} + best_fold_clusters.npy")
    log(f"PHASE 1 COMPLETE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(smoke=a.smoke)
