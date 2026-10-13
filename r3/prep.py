"""
R3 cohort preparation. Reuses the R2 CleanMLOmicsDataLoader + dedupe + preprocessor.

Modes:
  --cohort single_archive : keep only Clustering_datasets types (one processing pipeline; no
                            cross-archive batch axis). This is the decisive clean experiment (1B).
  --cohort all            : all 11 types (reproduces R2 cohort; for batch-correction 1C).

Saves cache to r3/cache_<cohort>/ : processed_data.npz, preprocessor.pkl, meta.pkl.
No result is tuned; this only prepares data.
"""
import os, sys, json, pickle, argparse, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "r2")); sys.path.insert(0, ROOT)
import nbdefs as nb
from clean_loader import CleanMLOmicsDataLoader, dedupe_and_relabel
from sklearn.preprocessing import LabelEncoder

config = nb.config


def main(cohort):
    cache = os.path.join(HERE, f"cache_{cohort}")
    os.makedirs(cache, exist_ok=True)
    loader = CleanMLOmicsDataLoader(config.data_path, config.data_variant,
                                    cache_path=os.path.join(ROOT, "r2", "cache", "clean_universe.pkl"))
    with contextlib.redirect_stdout(io.StringIO()):
        discovered = loader.discover_cancer_types()
    if cohort == "single_archive":
        discovered = [d for d in discovered if d[2] == "clustering"]
    print(f"[{cohort}] loading {len(discovered)} discovered archives: "
          f"{[(c, s) for c, _, s in discovered]}", flush=True)

    (omics, raw, surv, labels, sids, qr) = loader.load_pan_cancer(discovered, config.modalities)
    omics, raw, surv, labels, sids, dedup = dedupe_and_relabel(omics, raw, surv, labels, sids)
    print(f"[{cohort}] after dedup: {len(labels)} patients, types={sorted(set(map(str,labels)))}", flush=True)

    # batch label: 1 if the sample's data came from a classification archive, else 0.
    # After relabel, COAD came from the THCA (clustering) folder -> clustering. Classification-only
    # types are exactly those whose archive source was 'classification' at load time.
    src_by_type = {}
    for code, _, source in discovered:
        src_by_type.setdefault(code, source)
    # THCA relabelled to COAD -> clustering; ensure COAD marked clustering when single_archive
    def batch_of(lbl):
        # a type is 'classification' batch only if it was loaded solely from a GS- archive
        return 1 if src_by_type.get(lbl, "clustering") == "classification" and lbl != "COAD" else 0
    batch = np.array([batch_of(str(l)) for l in labels], dtype=int)

    config.num_clusters = len(set(map(str, labels)))
    pre = nb.MultiOmicsPreprocessor(config)
    processed = pre.fit_transform(omics)
    le = LabelEncoder(); ye = le.fit_transform(labels)

    N = len(labels)
    for m in config.modalities:
        assert processed[m].shape[0] == N and np.isfinite(processed[m]).all()

    np.savez_compressed(os.path.join(cache, "processed_data.npz"), **{m: processed[m] for m in config.modalities})
    with open(os.path.join(cache, "preprocessor.pkl"), "wb") as fh:
        pickle.dump(pre, fh)
    with open(os.path.join(cache, "meta.pkl"), "wb") as fh:
        pickle.dump(dict(cancer_labels=labels, sample_ids=sids, labels_encoded=ye, label_encoder=le,
                         survival_data=surv, quality_report=qr, num_clusters=config.num_clusters,
                         batch=batch, feature_dims={m: processed[m].shape[1] for m in config.modalities},
                         feature_names={m: list(pre.get_feature_names(m)) for m in config.modalities}), fh)
    print(f"[{cohort}] N={N} types={config.num_clusters} classification-batch samples={int(batch.sum())} "
          f"feature_dims={ {m: processed[m].shape[1] for m in config.modalities} }", flush=True)
    print(f"[{cohort}] cached to {cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True, choices=["single_archive", "all"])
    main(ap.parse_args().cohort)
