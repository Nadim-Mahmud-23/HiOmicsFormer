"""
PHASE 6 — External-validation scaffold (Reviewer 1.3).

Two deliverables:
  1. transform() refactor (already added to MultiOmicsPreprocessor in nbdefs) — a
     leakage-free projection of NEW data into the fitted TCGA feature space. This
     script self-tests it on a synthetic external cohort (no data reload needed).
  2. If an external cohort exists under data/external/, wire ExternalValidator per
     REVISION_PLAN Part 4; otherwise record NOT RUN with the exact call sequence.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import nbdefs as nb
from nbdefs import config
CACHE = os.path.join(HERE, "cache"); RES = config.results_path
EXT_DIR = os.path.join(ROOT, "data", "external")


def selftest_transform():
    with open(os.path.join(CACHE, "preprocessor.pkl"), "rb") as fh:
        pre = pickle.load(fh)
    rng = np.random.default_rng(0)
    fake, checks = {}, {}
    for mod in config.modalities:
        names = list(pre.get_feature_names(mod))
        # synthetic external cohort: 20 samples, columns = 80% of fitted names + 5 extras,
        # with 10% missing and one entirely-absent fitted feature
        keep = names[:int(0.8 * len(names))]
        cols = keep + [f"EXTRA_{i}" for i in range(5)]
        X = rng.normal(size=(20, len(cols)))
        df = pd.DataFrame(X, columns=cols)
        df.iloc[::5, 0] = np.nan                       # scattered missing
        fake[mod] = df
    out = pre.transform(fake)
    for mod in config.modalities:
        names = list(pre.get_feature_names(mod))
        arr = out[mod]
        checks[mod] = {
            "out_shape": list(arr.shape),
            "expected_cols": len(names),
            "shape_ok": bool(arr.shape == (20, len(names))),
            "finite_ok": bool(np.isfinite(arr).all()),
            # an absent fitted feature -> filled with fitted mean -> ~0 after z-scoring
            "absent_feature_is_zero": bool(abs(float(np.mean(arr[:, -1]))) < 1e-5
                                           if names[-1] not in fake[mod].columns else True),
        }
    ok = all(c["shape_ok"] and c["finite_ok"] for c in checks.values())
    return {"transform_selftest_passed": ok, "per_modality": checks}


def main():
    report = {"transform_refactor": "added to MultiOmicsPreprocessor.transform() (leakage-free)",
              "selftest": selftest_transform()}

    has_ext = os.path.isdir(EXT_DIR) and any(os.scandir(EXT_DIR))
    if has_ext:
        report["external_cohort"] = {"status": "PRESENT", "dir": EXT_DIR,
                                     "note": "wire ExternalValidator; see REVISION_PLAN Part 4"}
    else:
        report["external_cohort"] = {
            "status": "NOT RUN — external cohort not in repo",
            "expected_dir": EXT_DIR,
            "steps_to_complete": [
                "1. Place an independent multi-omics cohort with >=2 of {mRNA,miRNA,Methy,CNV} "
                "under data/external/ in the MLOmics layout (samples x features, HGNC symbols).",
                "   Candidates: TARGET (GDC), CPTAC (LinkedOmics), ICGC-ARGO.",
                "2. ext = CleanMLOmicsDataLoader(...).load_single_cancer(...)  # or custom loader",
                "3. aligned = ExternalValidator(model, preprocessor, config, device).align_features(ext)",
                "4. proc = preprocessor.transform(aligned)   # leakage-free (this refactor)",
                "5. build MultiOmicsDataset+DataLoader on proc; forward model(eval) -> z,q",
                "6. silhouette(z, argmax q); re-run AttentionBiomarkerExtractor -> Jaccard vs TCGA lists",
                "7. if external survival present: stratified_survival_analysis(z, ...).",
            ],
        }
    with open(os.path.join(RES, "external_validation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("PHASE 6:", "external cohort " + report["external_cohort"]["status"])
    print("  transform() self-test passed:", report["selftest"]["transform_selftest_passed"])


if __name__ == "__main__":
    main()
