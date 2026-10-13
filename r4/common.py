"""
R4 shared helpers (revision analyses requested by the reviewers).

All R4 scripts reuse the R2/R3 pipeline unchanged (r2/nbdefs.py = the notebook's model code,
r2/clean_loader.py = curation, r3/prep.py caches). Outputs go to results_revision/r4/.
Nothing here is tuned; every analysis is run once with fixed seeds.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
for p in (os.path.join(ROOT, "r3"), os.path.join(ROOT, "r2"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import nbdefs as nb  # noqa: E402
from nbdefs import config, device, SEED, HiOmicsFormer, MultiOmicsDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

OUT = os.path.join(ROOT, "results_revision", "r4")
os.makedirs(OUT, exist_ok=True)
EXT = os.path.join(ROOT, "data", "external")
CLASSIFICATION_TYPES = {"GBM", "LGG", "OV", "BRCA"}   # types present only in the MLOmics classification archive
# (BRCA enters only the orientation-corrected 13-type sensitivity cohort)


def log(msg):
    print(msg, flush=True)


def save_json(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"  wrote {path}")
    return path


def load_cache(cohort):
    """cohort in {'all', 'single_archive'} -> (processed dict, meta dict, fitted preprocessor)."""
    cache = os.path.join(ROOT, "r3", f"cache_{cohort}")
    d = np.load(os.path.join(cache, "processed_data.npz"))
    processed = {m: d[m] for m in config.modalities}
    with open(os.path.join(cache, "meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    with open(os.path.join(cache, "preprocessor.pkl"), "rb") as fh:
        pre = pickle.load(fh)
    return processed, meta, pre


def load_raw(cohort):
    """Re-load the curated, de-duplicated cohort BEFORE imputation/scaling, exactly as r3/prep.py
    does. Returns (omics dict of DataFrames with NaNs, labels array, sample_ids list, survival).
    cohort: 'all' | 'single_archive' (analysed cohorts) or 'all_fixed' | 'single_archive_fixed'
    (same, read with the orientation-corrected loader, which adds LUAD and BRCA)."""
    import io, contextlib
    from clean_loader import CleanMLOmicsDataLoader, OrientationFixedLoader, dedupe_and_relabel
    cls = OrientationFixedLoader if cohort.endswith("_fixed") else CleanMLOmicsDataLoader
    loader = cls(config.data_path, config.data_variant,
                 cache_path=os.path.join(ROOT, "r2", "cache", "clean_universe.pkl"))
    with contextlib.redirect_stdout(io.StringIO()):
        discovered = loader.discover_cancer_types()
        if cohort.startswith("single_archive"):
            discovered = [d for d in discovered if d[2] == "clustering"]
        omics, raw, surv, labels, sids, _ = loader.load_pan_cancer(discovered, config.modalities)
        omics, raw, surv, labels, sids, _ = dedupe_and_relabel(omics, raw, surv, labels, sids)
    return omics, np.asarray([str(x) for x in labels]), list(sids), surv


def load_model(path, feature_dims, num_clusters):
    config.num_clusters = num_clusters
    model = HiOmicsFormer(feature_dims, config).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def embed(model, processed):
    """Deterministic inference: returns (z = posterior mean, q = DEC soft assignments)."""
    mods = list(processed.keys())
    n = processed[mods[0]].shape[0]
    ds = MultiOmicsDataset(processed, np.zeros(n, dtype=int))
    zs, qs = [], []
    for b in DataLoader(ds, batch_size=config.batch_size, shuffle=False):
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        o = model(b)
        zs.append(o["z"].cpu().numpy())
        qs.append(o["q"].cpu().numpy())
    return np.concatenate(zs), np.concatenate(qs)


def archive_of(labels):
    """1 = classification archive (GBM/LGG/OV), 0 = clustering archive (incl. COAD from the THCA folder)."""
    return np.array([1 if str(l) in CLASSIFICATION_TYPES else 0 for l in labels], dtype=int)


def barcode(sample_id):
    """'ACC_TCGA.OR.A5J1.01' -> 'TCGA-OR-A5J1-01'."""
    bc = sample_id.split("_", 1)[1] if "_" in sample_id else sample_id
    return bc.replace(".", "-")


def association(clusters, variable):
    """Cramer's V (with and without singleton clusters), mutual information and the
    uncertainty coefficient U(variable | cluster) = I(C;Y)/H(Y). U is comparable across
    variables with different numbers of levels; Cramer's V is not."""
    from scipy.stats import chi2_contingency, entropy
    from sklearn.metrics import mutual_info_score
    clusters = np.asarray(clusters)
    variable = np.asarray([str(v) for v in variable])

    def v_of(c, y):
        t = pd.crosstab(c, y)
        if min(t.shape) < 2:
            return float("nan")
        chi2 = chi2_contingency(t, correction=False)[0]
        return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))

    sizes = pd.Series(clusters).value_counts()
    big = clusters[np.isin(clusters, sizes[sizes >= 10].index)]
    keep = np.isin(clusters, sizes[sizes >= 10].index)
    _, cnt = np.unique(variable, return_counts=True)
    mi = float(mutual_info_score(variable, clusters))
    return {"cramers_v": v_of(clusters, variable),
            "cramers_v_clusters_ge10": v_of(big, variable[keep]),
            "mutual_information_nats": mi,
            "uncertainty_coefficient": mi / float(entropy(cnt)) if len(cnt) > 1 else float("nan"),
            "n_levels": int(len(cnt))}
