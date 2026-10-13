"""
E2 - Baselines under the identical five-fold protocol, on either cohort  [Reviewer 1.2 / 1.7].

Re-runs the R2 baseline functions (r2/nbdefs.py) on the same StratifiedKFold(SEED) splits used for
HiOmicsFormer, with K = number of cancer types, and reports per-fold test-set silhouette / NMI / ARI.
Method names are reported as implemented:
  * 'MOGONET' in R2 was NOT a graph-convolutional network: it is an RBF multi-kernel similarity
    average followed by spectral clustering (unsupervised). It is renamed accordingly.
  * SNF is a re-implementation of the published fusion procedure (snfpy is incompatible with the
    installed scikit-learn), followed by spectral clustering.
Silhouettes are computed in each method's OWN representation space (as in R2) and are therefore
not comparable across methods; NMI/ARI against cancer type are.
Output: results_revision/r4/baselines_<cohort>.csv
"""
import argparse, os, time
import numpy as np
import pandas as pd
from common import ROOT, OUT, config, device, SEED, log, load_cache, MultiOmicsDataset
from nbdefs import run_simple_baselines, run_snf_baseline, run_mofa_baseline, run_mogonet_baseline
from hiomicsformer_revision_survival_ext import SimpleMultiOmicsTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch, json

RENAME = {"Concat + KMeans": "Concatenation + K-means", "MOGONET": "Multi-kernel spectral (RBF)",
          "mRNA Only": "mRNA only (K-means)", "miRNA Only": "miRNA only (K-means)",
          "Methy Only": "Methylation only (K-means)", "CNV Only": "CNV only (K-means)",
          "SimpleMultiOmicsTransformer": "Compact transformer + K-means"}


def transformer_fold(processed, labels, feature_dims, tr, te, ds, epochs):
    t = SimpleMultiOmicsTransformer(feature_dims, n_clusters=config.num_clusters, epochs=epochs,
                                    device=str(device), seed=SEED)
    t.fit_transform(DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(tr)))
    net = t.net; net.eval(); order, embs = [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(te)):
            order.extend(b["index"].numpy().tolist())
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            e, _ = net(b); embs.append(e.cpu().numpy())
    E = np.concatenate(embs); y = labels[np.array(order)]
    p = KMeans(n_clusters=config.num_clusters, random_state=SEED, n_init=10).fit_predict(E)
    return {"silhouette": float(silhouette_score(E, p)), "nmi": float(normalized_mutual_info_score(y, p)),
            "ari": float(adjusted_rand_score(y, p))}


def main(cohort, hio_dir, transformer_epochs):
    t0 = time.time()
    processed, meta, _ = load_cache(cohort)
    config.num_clusters = meta["num_clusters"]
    labels = meta["labels_encoded"]; N = len(labels)
    fd = {m: processed[m].shape[1] for m in config.modalities}
    ds = MultiOmicsDataset(processed, labels, sample_ids=meta["sample_ids"])
    res = {}
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    for k, (trv, te) in enumerate(skf.split(np.zeros(N), labels)):
        log(f"[{cohort}] fold {k+1}/{config.n_folds}")
        for name, m in run_simple_baselines(processed, labels, trv, te, config).items():
            res.setdefault(name, []).append(m)
        for name, fn in (("SNF", run_snf_baseline), ("MOFA+", run_mofa_baseline), ("MOGONET", run_mogonet_baseline)):
            m = fn(processed, labels, trv, te, config)
            if m:
                res.setdefault(name, []).append(m)
        res.setdefault("SimpleMultiOmicsTransformer", []).append(
            transformer_fold(processed, labels, fd, trv, te, ds, transformer_epochs))
    rows = []
    for name, lst in res.items():
        r = {"method": RENAME.get(name, name), "n_folds": len(lst)}
        for met in ("silhouette", "nmi", "ari"):
            v = [float(x[met]) for x in lst]
            r[f"{met}_mean"], r[f"{met}_sd"] = float(np.mean(v)), float(np.std(v))
        rows.append(r)
    cv = json.load(open(os.path.join(ROOT, "results_revision", hio_dir, "cv_metrics.json")))["cv_summary"]
    rows.append({"method": "HiOmicsFormer", "n_folds": len(cv["silhouette"]["values"]),
                 **{f"{m}_mean": cv[m]["mean"] for m in ("silhouette", "nmi", "ari")},
                 **{f"{m}_sd": cv[m]["std"] for m in ("silhouette", "nmi", "ari")}})
    df = pd.DataFrame(rows).sort_values("nmi_mean", ascending=False)
    path = os.path.join(OUT, f"baselines_{cohort}.csv"); df.to_csv(path, index=False)
    log(df.to_string(index=False)); log(f"wrote {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="single_archive", choices=["single_archive", "all"])
    ap.add_argument("--hio-dir", default=None, help="results_revision/<dir> holding HiOmicsFormer cv_metrics.json")
    ap.add_argument("--transformer-epochs", type=int, default=50)
    a = ap.parse_args()
    main(a.cohort, a.hio_dir or ("single_archive" if a.cohort == "single_archive" else "all_uncorrected"),
         a.transformer_epochs)
