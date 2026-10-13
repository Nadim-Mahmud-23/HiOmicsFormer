"""
Rigor diagnostic for Reviewer 1.5: WHY is the fixed imputation Jaccard ~1.0 (and is it
degenerate)? Recomputes, per modality, the top-50 MAD features and reports:
  * per-feature missingness of the selected top-50 (low missing -> imputation can't move them)
  * MAD separation at the rank-50/51 boundary (well-separated -> stable ranking, not a tie)
  * that the ranking is NON-degenerate (unlike variance-after-StandardScaler).
Writes results_revision/imputation_rigor.json.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import nbdefs as nb
from clean_loader import CleanMLOmicsDataLoader, dedupe_and_relabel
from sklearn.impute import SimpleImputer
config = nb.config


def main():
    ld = CleanMLOmicsDataLoader(config.data_path, config.data_variant,
                                cache_path=os.path.join(HERE, "cache", "clean_universe.pkl"))
    disc = ld.discover_cancer_types()
    omics, raw, surv, labels, sids, qr = ld.load_pan_cancer(disc, config.modalities)
    omics, raw, surv, labels, sids, _ = dedupe_and_relabel(omics, raw, surv, labels, sids)

    out = {}
    for mod in config.modalities:
        X = raw[mod].values.astype(np.float32)
        names = list(raw[mod].columns)
        # same MAD selection as imputation_sensitivity_fixed
        mad_all = np.nanmedian(np.abs(X - np.nanmedian(X, axis=0)), axis=0)
        k = min(config.feature_selection_top_k, X.shape[1])
        sel = np.argsort(np.nan_to_num(mad_all, nan=-np.inf))[-k:]
        Xs = X[:, sel]; ns = [names[i] for i in sel]
        keepf = np.isnan(Xs).mean(0) < 0.90
        Xs = Xs[:, keepf]; ns = [n for n, kf in zip(ns, keepf) if kf]
        Xs = Xs[np.isnan(Xs).mean(1) < 0.95]
        # mean-impute, MAD rank
        Xi = SimpleImputer(strategy="mean").fit_transform(Xs)
        Xi = np.nan_to_num(Xi)
        mad = np.median(np.abs(Xi - np.median(Xi, 0)), 0)
        order = np.argsort(mad)[::-1]
        top = order[:50]
        miss = np.isnan(Xs).mean(0)                     # per-feature missingness (pre-impute)
        mad_sorted = np.sort(mad)[::-1]
        gap = float(mad_sorted[49] - mad_sorted[50]) if len(mad_sorted) > 50 else None
        out[mod] = {
            "top50_missingness_median": float(np.median(miss[top])),
            "top50_missingness_max": float(np.max(miss[top])),
            "overall_selected_missingness_median": float(np.median(miss)),
            "mad_rank50_value": float(mad_sorted[49]),
            "mad_rank51_value": float(mad_sorted[50]) if len(mad_sorted) > 50 else None,
            "mad_boundary_gap": gap,
            "n_zero_mad_features": int((mad == 0).sum()),
            "degenerate_all_equal": bool(np.allclose(mad, mad[0])),
        }
        print(f"{mod:6s}: top50 missing median={out[mod]['top50_missingness_median']:.1%} "
              f"max={out[mod]['top50_missingness_max']:.1%}; boundary gap={gap:.3f}; "
              f"degenerate={out[mod]['degenerate_all_equal']}", flush=True)
    json.dump(out, open(os.path.join(config.results_path, "imputation_rigor.json"), "w"), indent=2)
    print("wrote imputation_rigor.json")


if __name__ == "__main__":
    main()
