"""
(1) Scale signature  [audit M2]: after pooling and standardisation, how much do the values the model
    receives vary WITHIN each cancer type, per modality? Files that entered on a different scale
    (beta values, unscaled CNV) are compressed to near-constant values after pooled z-scoring - a
    technical signature that biological differences between tumour types do not produce. Also: which
    modalities drive the input's first principal component.
(2) Residual type information  [audit M3]: cross-validated supervised classification of cancer type
    from the single-archive input (files z-scored within type) and from the full-cohort input.
    Balanced accuracy; chance = 1/number of types.
Output: results_revision/r4/scale_signature.json
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from common import SEED, config, log, save_json, load_cache


def main():
    res = {}
    processed, meta, _ = load_cache("all")
    lab = np.asarray([str(x) for x in meta["cancer_labels"]])
    sd = {}
    for m in config.modalities:
        X = processed[m]
        sd[m] = {t: float(np.median(X[lab == t].std(axis=0))) for t in sorted(set(lab))}
    res["median_within_type_feature_sd"] = sd
    grp = np.isin(lab, ["GBM", "LGG", "OV"])
    res["median_within_type_sd_summary"] = {
        m: {"GBM_LGG_OV": float(np.median([sd[m][t] for t in ("GBM", "LGG", "OV")])),
            "other_types": float(np.median([v for t, v in sd[m].items() if t not in ("GBM", "LGG", "OV")]))}
        for m in config.modalities}
    X = np.concatenate([processed[m] for m in config.modalities], axis=1)
    pca = PCA(1, random_state=SEED).fit(X)
    load2 = pca.components_[0] ** 2; bounds = np.cumsum([0] + [processed[m].shape[1] for m in config.modalities])
    res["input_pc1_loading_share"] = {m: float(load2[bounds[i]:bounds[i + 1]].sum()) for i, m in enumerate(config.modalities)}
    log(f"within-type SD summary: {res['median_within_type_sd_summary']}; PC1 loading share: {res['input_pc1_loading_share']}")

    clfs = {"logistic regression (100 PCs)": make_pipeline(PCA(100, random_state=SEED), LogisticRegression(max_iter=3000)),
            "k-nearest neighbours (50 PCs, k=15)": make_pipeline(PCA(50, random_state=SEED), KNeighborsClassifier(15)),
            "random forest (100 PCs, 300 trees)": make_pipeline(PCA(100, random_state=SEED), RandomForestClassifier(300, random_state=SEED, n_jobs=-1))}
    res["supervised_balanced_accuracy"] = {}
    for cohort in ("single_archive", "all"):
        proc, mt, _ = load_cache(cohort)
        y = np.asarray([str(x) for x in mt["cancer_labels"]])
        Xc = np.concatenate([proc[m] for m in config.modalities], axis=1)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        out = {"chance": 1.0 / len(set(y))}
        for name, clf in clfs.items():
            pred = cross_val_predict(clf, Xc, y, cv=cv)
            out[name] = float(balanced_accuracy_score(y, pred))
        res["supervised_balanced_accuracy"][cohort] = out
        log(f"[{cohort}] {out}")
    save_json(res, "scale_signature.json")


if __name__ == "__main__":
    main()
