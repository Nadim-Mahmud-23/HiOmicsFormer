"""
Data audit: value scale of every MLOmics 'Original' file (pinned release), read directly from the
CSVs (features x samples), plus the duplicate-COAD identity test.

Finding this documents: most files are z-scored per feature WITHIN the file (gene means 0, SDs ~1),
which the MLOmics paper does not describe for the Original version, whereas GS-GBM, GS-LGG and GS-OV
are on other scales (methylation as beta values in [0, 1]; CNV unscaled; many constant miRNA rows).
Pooling such files makes 'which file a patient came from' the dominant source of variance, and
per-file z-scoring removes between-cancer-type mean differences.
Output: results_revision/r4/normalization_audit.csv, duplicate_coad_identity.json
"""
import glob, os
import numpy as np
import pandas as pd
from common import ROOT, OUT, config, log, save_json

BASE = os.path.join(ROOT, "data", "MLOmics", "Main_Dataset")


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, "Clustering_datasets", "*", "Original", "*_*.csv")) +
                    glob.glob(os.path.join(BASE, "Classification_datasets", "GS-*", "Original", "*_*.csv"))):
        name = os.path.basename(f)
        if name.startswith("survival") or "label" in name:
            continue
        X = pd.read_csv(f, index_col=0).values.astype(float)        # features x samples
        gm, gs = np.nanmean(X, axis=1), np.nanstd(X, axis=1)
        archive = "clustering" if "Clustering_datasets" in f else "classification"
        folder = f.split(os.sep)[-3]
        zscored = bool(np.nanmedian(np.abs(gm)) < 0.05 and abs(np.nanmedian(gs) - 1) < 0.05)
        rows.append({"archive": archive, "folder": folder, "modality": name.split("_")[-1].replace(".csv", ""),
                     "features": X.shape[0], "samples": X.shape[1],
                     "median_abs_feature_mean": float(np.nanmedian(np.abs(gm))), "median_feature_sd": float(np.nanmedian(gs)),
                     "frac_constant_features": float(np.mean(gs < 1e-8)), "min": float(np.nanmin(X)), "max": float(np.nanmax(X)),
                     "zscored_within_file": zscored})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "normalization_audit.csv"), index=False)
    log(df.round(3).to_string(index=False))

    # duplicate COAD copies: same data up to per-file standardisation?
    res = {}
    for m in config.modalities:
        a = pd.read_csv(os.path.join(BASE, "Clustering_datasets", "THCA", "Original", f"THCA_{m}.csv"), index_col=0)
        b = pd.read_csv(os.path.join(BASE, "Classification_datasets", "GS-COAD", "Original", f"COAD_{m}.csv"), index_col=0)
        a.index = a.index.str.replace(".", "-", regex=False); b.index = b.index.str.replace(".", "-", regex=False)
        feats, samp = a.index.intersection(b.index), a.columns.intersection(b.columns)
        A, B = a.loc[feats, samp].values.astype(float), b.loc[feats, samp].values.astype(float)
        ok = (A.std(1) > 0) & (B.std(1) > 0)
        r = np.array([np.corrcoef(A[i], B[i])[0, 1] for i in np.where(ok)[0]])
        Az = (A - A.mean(1, keepdims=True)) / A.std(1, ddof=1, keepdims=True)
        d = np.abs(Az - B)[ok]
        res[m] = {"shared_features": int(len(feats)), "shared_patients": int(len(samp)),
                  "per_feature_corr_min": float(r.min()), "per_feature_corr_median": float(np.median(r)),
                  "restandardised_abs_diff_median": float(np.median(d)), "restandardised_abs_diff_max": float(d.max())}
    log(res)
    save_json(res, "duplicate_coad_identity.json")


def symbol_audit():
    """Spreadsheet date conversions among feature names of the curated universe and the selected panels."""
    import pickle, re
    uni = pickle.load(open(os.path.join(ROOT, "r2", "cache", "clean_universe.pkl"), "rb"))
    mon = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    num = re.compile(r"^\d+(\.\d+)?$"); dat = re.compile(r"^\d{1,2}-" + mon + r"$|^" + mon + r"-\d{1,2}$")
    from common import load_cache
    _, _, pre = load_cache("all")
    out = {}
    for m in config.modalities:
        u = uni[m]; sel = [str(x) for x in pre.get_feature_names(m)]
        out[m] = {"universe_numeric_serials": sum(bool(num.match(x)) for x in u),
                  "universe_date_strings": sum(bool(dat.match(x)) for x in u),
                  "selected_numeric_serials": sum(bool(num.match(x)) for x in sel),
                  "selected_date_strings": sum(bool(dat.match(x)) for x in sel),
                  "examples": sorted([x for x in u if num.match(x) or dat.match(x)])[:6]}
        out[m]["universe_total_converted"] = out[m]["universe_numeric_serials"] + out[m]["universe_date_strings"]
    log(out)
    save_json(out, "symbol_audit.json")


if __name__ == "__main__":
    main()
    symbol_audit()
