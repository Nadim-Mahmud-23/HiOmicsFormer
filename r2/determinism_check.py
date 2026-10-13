"""
PHASE 7 determinism check (Reviewer 6.2 / brief Phase 7).

Trains ONE fold TWICE with identical seeding and confirms the resulting test
silhouette + embeddings are identical (rules out nondeterminism). Runs in
isolation — writes only results_revision/determinism_check.json, never touches the
real Phase-1 outputs. Uses a short schedule (identical-seed runs that are going to
diverge diverge within a few epochs).
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import torch
import nbdefs as nb
from nbdefs import (config, device, SEED, HiOmicsFormer, HiOmicsLoss, HiOmicsTrainer,
                    HiOmicsConfig, MultiOmicsDataset, init_centroids)
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import silhouette_score

CACHE = os.path.join(HERE, "cache")


def one_fold():
    # reseed EXACTLY as the notebook does at import
    np.random.seed(SEED); torch.manual_seed(SEED)
    try:
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    d = np.load(os.path.join(CACHE, "processed_data.npz"))
    processed = {m: d[m] for m in config.modalities}
    meta = pickle.load(open(os.path.join(CACHE, "meta.pkl"), "rb"))
    config.num_clusters = meta["num_clusters"]
    labels = meta["labels_encoded"]
    ds = MultiOmicsDataset(processed, labels, sample_ids=meta["sample_ids"])
    fdims = ds.get_feature_dims()
    N = len(labels)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    tr_val, te = next(iter(skf.split(np.zeros(N), labels)))
    tr, va = train_test_split(tr_val, test_size=0.2, stratify=labels[tr_val], random_state=SEED)

    def mk(idx): return DataLoader(ds, batch_size=config.batch_size,
                                   sampler=SubsetRandomSampler(idx), num_workers=0)
    trL, vaL, teL = mk(tr), mk(va), mk(te)
    m = HiOmicsFormer(fdims, config).to(device)

    c1 = HiOmicsConfig(); c1.num_clusters = config.num_clusters
    c1.lambda_DEC = 0.0; c1.max_epochs = 8; c1.patience = 8
    opt = torch.optim.AdamW(m.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    HiOmicsTrainer(m, opt, HiOmicsLoss(c1), None, c1, device).train_fold(
        trL, vaL, labels[tr], labels[va])
    init_centroids(m, trL, config, device)
    c2 = HiOmicsConfig(); c2.num_clusters = config.num_clusters; c2.max_epochs = 8; c2.patience = 8
    opt2 = torch.optim.AdamW(m.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    t2 = HiOmicsTrainer(m, opt2, HiOmicsLoss(c2), None, c2, device)
    t2.train_fold(trL, vaL, labels[tr], labels[va])

    m.eval(); zs, ps = [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=config.batch_size, sampler=list(te)):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o = m(b); zs.append(o["z"].cpu().numpy()); ps.append(o["q"].argmax(1).cpu().numpy())
    z = np.concatenate(zs); p = np.concatenate(ps)
    sil = float(silhouette_score(z, p)) if len(np.unique(p)) > 1 else 0.0
    return sil, z


def main():
    print("determinism: training fold-0 twice with identical seed...", flush=True)
    s1, z1 = one_fold()
    s2, z2 = one_fold()
    maxdiff = float(np.abs(z1 - z2).max())
    out = {"silhouette_run1": s1, "silhouette_run2": s2,
           "silhouette_identical": bool(s1 == s2),
           "embeddings_maxdiff": maxdiff,
           "embeddings_close_atol_1e-4": bool(np.allclose(z1, z2, atol=1e-4)),
           "device": str(device),
           "note": ("Identical seeds -> identical silhouette confirms determinism. On MPS, "
                    "tiny (<1e-4) float differences may remain from nondeterministic GPU "
                    "reductions; silhouette equality is the operative check.")}
    json.dump(out, open(os.path.join(config.results_path, "determinism_check.json"), "w"), indent=2)
    print(f"  run1 sil={s1:.6f}  run2 sil={s2:.6f}  identical={out['silhouette_identical']}  "
          f"emb_maxdiff={maxdiff:.2e}", flush=True)


if __name__ == "__main__":
    main()
