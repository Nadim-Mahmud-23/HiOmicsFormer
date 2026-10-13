"""
R4 reproduction check: the re-downloaded MLOmics release (Hugging Face AIBIC/MLOmics, commit
d7c2239504dcf817ee9f33a693f9617d614eef67) must reproduce the R3 caches exactly.
  1. rebuilt meta/preprocessor (r3/prep.py) == the R3 copies backed up in results_revision/r4/r3_cache_backup/
     (same patients, labels, archive flags and selected feature names per modality);
  2. the saved single-archive model, applied to the rebuilt processed data, reproduces the stored
     single-archive embeddings (deterministic inference).
Output: results_revision/r4/reproduction_check.json
"""
import os, pickle
import numpy as np
from common import ROOT, OUT, config, log, save_json, load_cache, load_model, embed


class _Stub:
    def __setstate__(self, s):
        self.__dict__.update(s)


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod in ("nbdefs", "__main__") and name == "MultiOmicsPreprocessor":
            return _Stub
        return super().find_class(mod, name)


def main():
    rep = {}
    for cohort in ("single_archive", "all"):
        bdir = os.path.join(OUT, "r3_cache_backup", cohort)
        old_meta = pickle.load(open(os.path.join(bdir, "meta.pkl"), "rb"))
        old_pre = _U(open(os.path.join(bdir, "preprocessor.pkl"), "rb")).load()
        processed, meta, pre = load_cache(cohort)
        r = {"same_sample_ids": list(old_meta["sample_ids"]) == list(meta["sample_ids"]),
             "same_labels": [str(x) for x in old_meta["cancer_labels"]] == [str(x) for x in meta["cancer_labels"]],
             "same_batch": bool(np.array_equal(old_meta["batch"], meta["batch"])),
             "same_selected_features": {m: list(old_pre.feature_names[m]) == list(pre.feature_names[m]) for m in config.modalities},
             "same_scaler_means": {m: bool(np.allclose(old_pre.scalers[m].mean_, pre.scalers[m].mean_)) for m in config.modalities},
             "n": len(meta["sample_ids"])}
        rep[cohort] = r
        log(f"[{cohort}] {r}")
    processed, meta, _ = load_cache("single_archive")
    fd = {m: processed[m].shape[1] for m in config.modalities}
    model = load_model(os.path.join(ROOT, "results_revision", "single_archive", "best_model.pt"), fd, meta["num_clusters"])
    z, q = embed(model, processed)
    stored = np.load(os.path.join(ROOT, "results_revision", "single_archive", "embeddings.npy"))
    rep["single_archive_embedding_max_abs_diff"] = float(np.abs(z - stored).max())
    rep["single_archive_clusters_identical"] = bool(np.array_equal(q.argmax(1), np.load(
        os.path.join(ROOT, "results_revision", "single_archive", "clusters.npy"))))
    log(f"single-archive re-embedding: max|diff|={rep['single_archive_embedding_max_abs_diff']:.2e}, "
        f"clusters identical={rep['single_archive_clusters_identical']}")
    save_json(rep, "reproduction_check.json")


if __name__ == "__main__":
    main()
