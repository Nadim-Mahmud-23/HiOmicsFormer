"""
Phase 1D — DEC collapse sweep. Single train/val/test split (not 5-fold, for tractability) on the
single-archive cohort. Sweeps K_c and lambda_DEC, scoring test silhouette, NMI, ARI, effective
clusters, and utilization (= effective/K_c). K_c is later chosen by a STATED rule (utilization-
weighted, not by prettiest downstream story). Writes results_revision/dec_collapse_sweep.csv.

Nothing tuned toward a target; this is a diagnostic sweep whose full table is reported.
Usage: python3 r3/dec_sweep.py --cohort single_archive [--epochs 40]
"""
import os, sys, argparse, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "r2")); sys.path.insert(0, ROOT)
import pickle, torch
import nbdefs as nb
from nbdefs import (config, device, SEED, HiOmicsFormer, HiOmicsLoss, HiOmicsTrainer,
                    HiOmicsConfig, MultiOmicsDataset, init_centroids)
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, normalized_mutual_info_score as NMI, adjusted_rand_score as ARI


def aligned_eval(model, ds, idx):
    model.eval(); zs, ps, ii = [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(idx)):
            j = b["index"]; b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o = model(b); zs.append(o["z"].cpu().numpy()); ps.append(o["q"].argmax(1).cpu().numpy())
            ii.append(j.numpy())
    z = np.concatenate(zs); p = np.concatenate(ps); o = np.argsort(np.concatenate(ii))
    return z[o], p[o]


def run_one(processed, labels, sids, Kc, lam_dec, epochs):
    fd = {m: processed[m].shape[1] for m in config.modalities}
    ds = MultiOmicsDataset(processed, labels, sample_ids=sids)
    N = len(labels)
    tr, te = train_test_split(np.arange(N), test_size=0.2, stratify=labels, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.2, stratify=labels[tr], random_state=SEED)
    cfg = HiOmicsConfig(); cfg.num_clusters = Kc; cfg.lambda_DEC = lam_dec; cfg.max_epochs = epochs; cfg.patience = 15
    mk = lambda i: DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(i))
    model = HiOmicsFormer(fd, cfg).to(device)
    c1 = HiOmicsConfig(); c1.num_clusters = Kc; c1.lambda_DEC = 0.0; c1.max_epochs = 20; c1.patience = 10
    o1 = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    HiOmicsTrainer(model, o1, HiOmicsLoss(c1), None, c1, device)  # build; run below
    tr1 = HiOmicsTrainer(model, o1, HiOmicsLoss(c1), None, c1, device)
    import types
    # minimal train loop (no eval dependency)
    for _ in range(c1.max_epochs):
        model.train()
        for b in mk(tr):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o1.zero_grad(); out = model(b); loss, _ = HiOmicsLoss(c1)(b, out, model, 99); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); o1.step()
    with torch.no_grad():
        cfg.num_clusters = Kc
    init_centroids(model, mk(tr), cfg, device)
    o2 = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for _ in range(epochs):
        model.train()
        for b in mk(tr):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o2.zero_grad(); out = model(b); loss, _ = HiOmicsLoss(cfg)(b, out, model, 99); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); o2.step()
    z, p = aligned_eval(model, ds, te)
    y = labels[np.sort(te)] if False else labels[te]
    # align y to sorted-te order
    y = labels[np.sort(te)]
    eff = int((np.bincount(p, minlength=Kc) > 0).sum())
    sil = float(silhouette_score(z, p)) if len(np.unique(p)) > 1 else 0.0
    return {"Kc": Kc, "lambda_DEC": lam_dec, "test_silhouette": round(sil, 3),
            "test_nmi": round(float(NMI(y, p)), 3), "test_ari": round(float(ARI(y, p)), 3),
            "effective_clusters": eff, "utilization": round(eff / Kc, 3)}


def main(cohort, epochs):
    cache = os.path.join(HERE, f"cache_{cohort}")
    d = np.load(os.path.join(cache, "processed_data.npz")); processed = {m: d[m] for m in config.modalities}
    meta = pickle.load(open(os.path.join(cache, "meta.pkl"), "rb"))
    labels = meta["labels_encoded"]; sids = meta["sample_ids"]
    rows = []
    for Kc in [2, 3, 4, 5, 6, 8, 10, 12, 15]:
        r = run_one(processed, labels, sids, Kc, 0.5, epochs)
        rows.append(r); print(r, flush=True)
    # a couple of lambda_DEC variants at a mid Kc to see if weaker DEC reduces collapse
    for lam in [0.1, 1.0]:
        r = run_one(processed, labels, sids, 8, lam, epochs); r["note"] = "lambda_sweep"; rows.append(r); print(r, flush=True)
    out = "results_revision/dec_collapse_sweep.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["note"]); w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", default="single_archive"); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(); main(a.cohort, a.epochs)
