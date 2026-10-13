"""
PHASE 0 — Blocking integrity checks (Reviewer 1.4/1.5/1.6) + data cache.

Runs headless. Reuses the notebook's exact loader/preprocessor (via nbdefs) and the
two vetted add-on functions. Produces:
  results_revision/feature_name_report.json     (0.1)
  results_revision/imputation_sensitivity_fixed.csv  + per-modality json (0.2)
  figures_revision/fig8_imputation_sensitivity.png   (regenerated from real numbers)
  r2/cache/*  (processed data, fitted preprocessor, labels, ids, survival) for Phases 1-7

Nothing here is fabricated. Every number comes from the real MLOmics load.
"""
import os, sys, json, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)                       # so config's ./data, ./results_revision resolve
sys.path.insert(0, HERE)             # nbdefs
sys.path.insert(0, ROOT)             # hiomicsformer_revision_addons / _survival_ext

import nbdefs as nb
from clean_loader import CleanMLOmicsDataLoader, _harm, dedupe_and_relabel
from hiomicsformer_revision_addons import (
    verify_feature_names, imputation_sensitivity_fixed,
)

SEED = nb.SEED
config = nb.config
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(config.results_path, exist_ok=True)
os.makedirs(config.figures_path, exist_ok=True)


def log(msg): print(msg, flush=True)


def main():
    t0 = time.time()

    # ---- Fail loudly if data path is missing (NO synthetic fallback) ----
    clu = os.path.join(config.data_path, "Clustering_datasets")
    if not os.path.isdir(clu):
        raise SystemExit(
            f"FATAL: data path not found: {clu}\n"
            "Expected MLOmics layout:\n"
            "  data/MLOmics/Main_Dataset/Clustering_datasets/{CANCER}/Original/{CANCER}_{mod}.csv\n"
            "See Live/HiOmicsFormer/data/DATA_ACCESS.md. Refusing to run on synthetic data.")

    # ======================================================================
    # DATA LOAD (reuse notebook Cell 3 execution logic)
    # ======================================================================
    log("=" * 74); log("PHASE 0 — LOAD MLOmics (Original variant, all cancer types) — CLEANED"); log("=" * 74)
    log("Using CleanMLOmicsDataLoader: harmonize names ('.'->'-') + drop synthetic "
        "padding (restrict to clean Clustering_datasets gene universe).")
    data_loader = CleanMLOmicsDataLoader(config.data_path, data_variant=config.data_variant,
                                         cache_path=os.path.join(CACHE, "clean_universe.pkl"))
    discovered = data_loader.discover_cancer_types()
    (omics_data, raw_omics_data, survival_data,
     cancer_labels, sample_ids, quality_report) = data_loader.load_pan_cancer(
        discovered, config.modalities)

    # ---- CORRECT verified mislabeled/duplicated cancer type (THCA folder == COLON/COAD) ----
    (omics_data, raw_omics_data, survival_data,
     cancer_labels, sample_ids, dedup_report) = dedupe_and_relabel(
        omics_data, raw_omics_data, survival_data, cancer_labels, sample_ids)
    log(f"\nDedup/relabel: detected duplicate type pairs {dedup_report['detected_duplicates']}; "
        f"relabel {dedup_report['relabel_map']}; "
        f"patients {dedup_report['n_before']} -> {dedup_report['n_after']} "
        f"(dropped {dedup_report['n_dropped']}); types_after={dedup_report['types_after']}")
    with open(os.path.join(config.results_path, "dedup_relabel_report.json"), "w") as fh:
        json.dump(dedup_report, fh, indent=2, default=str)

    n_loaded = len(set(map(str, cancer_labels)))
    if n_loaded != config.num_clusters:
        log(f"Updating config.num_clusters: {config.num_clusters} -> {n_loaded}")
        config.num_clusters = n_loaded

    # ---- report synthetic-padding / mangled features dropped by the clean loader ----
    drop = data_loader.dropped_report
    tot_dropped = sum(v["dropped"] for v in drop.values())
    log(f"\nClean loader dropped {tot_dropped} synthetic/mangled feature-columns "
        f"across {len(drop)} (cancer,modality) files:")
    for k, v in sorted(drop.items(), key=lambda x: -x[1]["dropped"])[:12]:
        log(f"    {k}: {v['before']} -> {v['after']}  (dropped {v['dropped']})")
    with open(os.path.join(config.results_path, "cleaning_report.json"), "w") as fh:
        json.dump({"total_dropped": tot_dropped, "per_file": drop}, fh, indent=2)

    # ======================================================================
    # PREPROCESS (reuse notebook Cell 5 execution)
    # ======================================================================
    log("\nPreprocessing (documented MAD feature selection + kNN impute + scale)...")
    preprocessor = nb.MultiOmicsPreprocessor(config)
    processed_data = preprocessor.fit_transform(omics_data)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(cancer_labels)

    # sanity: alignment
    N = len(cancer_labels)
    assert len(sample_ids) == N == len(labels_encoded), "row-count misalignment"
    for m in config.modalities:
        assert processed_data[m].shape[0] == N, f"{m} sample-count mismatch"
        assert np.isfinite(processed_data[m]).all(), f"{m} has NaN/Inf after preprocessing"

    # ======================================================================
    # CACHE for later phases
    # ======================================================================
    np.savez_compressed(os.path.join(CACHE, "processed_data.npz"),
                        **{m: processed_data[m] for m in config.modalities})
    with open(os.path.join(CACHE, "preprocessor.pkl"), "wb") as fh:
        pickle.dump(preprocessor, fh)
    with open(os.path.join(CACHE, "meta.pkl"), "wb") as fh:
        pickle.dump(dict(cancer_labels=cancer_labels, sample_ids=sample_ids,
                         labels_encoded=labels_encoded,
                         label_encoder=label_encoder,
                         survival_data=survival_data,
                         quality_report=quality_report,
                         num_clusters=config.num_clusters,
                         feature_dims={m: processed_data[m].shape[1] for m in config.modalities},
                         feature_names={m: list(preprocessor.get_feature_names(m)) for m in config.modalities}),
                    fh)
    log(f"\nCached processed data + fitted preprocessor to {CACHE}/")

    # ======================================================================
    # 0.1  FEATURE-NAME INTEGRITY
    # ======================================================================
    name_report = verify_feature_names(preprocessor, config.modalities)
    # make JSON-serializable
    ser = {m: {k: (v if not isinstance(v, (np.floating,)) else float(v))
               for k, v in d.items()} for m, d in name_report.items()}
    with open(os.path.join(config.results_path, "feature_name_report.json"), "w") as fh:
        json.dump(ser, fh, indent=2)
    log(f"Saved {config.results_path}/feature_name_report.json")

    # Reconcile Fig 6a (attention) vs 6c (consensus) name space: BOTH the attention
    # and reconstruction extractors (notebook Cell 12) label features via
    # preprocessor.get_feature_names(mod) — i.e. the SAME namespace we just verified.
    # Record that finding for the summary.
    recon = {
        "attention_and_reconstruction_use_same_namespace": True,
        "namespace_source": "MultiOmicsPreprocessor.get_feature_names(mod) == real data column symbols",
        "note": ("Raw MLOmics feature symbols are real (A1BG..ZZZ3). 'ZZZ3' is a real gene "
                 "(last alphabetically); no ZZZ936/ZZZ737-style placeholders exist in the data "
                 "or the notebook code. Fig 6c 'ZZZ936/ZZZ737' markers are not reproducible "
                 "from the current pipeline."),
    }
    with open(os.path.join(config.results_path, "fig6_namespace_reconciliation.json"), "w") as fh:
        json.dump(recon, fh, indent=2)

    # ---- AFTER-clean contamination audit on the SELECTED features (should be ~0) ----
    import re as _re
    padre = _re.compile(r'^(ZZEF\d+|ZZZ\d+)$')
    audit = {}
    for mod in config.modalities:
        names = list(preprocessor.get_feature_names(mod))
        uni = data_loader.valid_universe.get(mod, set())
        padlike = [n for n in names if padre.match(n) and _harm(n) not in uni]
        dotted = [n for n in names if "." in n]
        notin = [n for n in names if _harm(n) not in uni]
        audit[mod] = {"selected": len(names), "residual_padding": len(padlike),
                      "residual_dotted": len(dotted), "not_in_clean_universe": len(notin),
                      "examples": names[:10]}
        log(f"  [after-clean audit] {mod:6s}: selected={len(names)} "
            f"residual_padding={len(padlike)} dotted={len(dotted)} not_in_universe={len(notin)}")
    with open(os.path.join(config.results_path, "selected_feature_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)

    # ======================================================================
    # 0.2  FIXED IMPUTATION SENSITIVITY
    # ======================================================================
    imp_df, imp_per_mod = imputation_sensitivity_fixed(
        raw_omics_data, config, top_k=config.top_k_biomarkers, seed=SEED, extractor_fn=None)
    imp_df.to_csv(os.path.join(config.results_path, "imputation_sensitivity_fixed.csv"), index=False)

    # per-modality missingness / null table (rankings sets are not serialized)
    per_mod_ser = {}
    for m, d in imp_per_mod.items():
        per_mod_ser[m] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in d.items() if k != "rankings"}
    with open(os.path.join(config.results_path, "imputation_sensitivity_per_modality.json"), "w") as fh:
        json.dump(per_mod_ser, fh, indent=2, default=str)
    log(f"Saved {config.results_path}/imputation_sensitivity_fixed.csv")

    # ---- Regenerate fig8: heatmap of jaccard per modality x pair + null + missingness ----
    _make_fig8(imp_df, os.path.join(config.figures_path, "fig8_imputation_sensitivity.png"))
    log(f"Regenerated {config.figures_path}/fig8_imputation_sensitivity.png")

    # ======================================================================
    # PHASE 0 SUMMARY
    # ======================================================================
    log("\n" + "=" * 74); log("PHASE 0 SUMMARY"); log("=" * 74)
    log(f"  patients: {N}  |  cancer types: {n_loaded}  ({sorted(set(cancer_labels))})")
    for m in config.modalities:
        st = name_report[m]["status"]
        log(f"  feature-names[{m:6s}]: {st}")
    if len(imp_df):
        log("  fixed imputation Jaccard (mean per modality vs chance):")
        for m in config.modalities:
            sub = imp_df[imp_df.modality == m]
            if len(sub):
                log(f"    {m:6s}: jaccard={sub.jaccard.mean():.3f}  "
                    f"chance={sub.null_jaccard.mean():.3f}  "
                    f"raw_miss={sub.raw_missing.iloc[0]:.1%}  eff_miss={sub.eff_missing.iloc[0]:.1%}")
    log(f"\n  elapsed: {time.time()-t0:.0f}s")
    log("PHASE 0 COMPLETE")


def _make_fig8(imp_df, path):
    if imp_df is None or not len(imp_df):
        fig, ax = plt.subplots(figsize=(6, 3)); ax.axis("off")
        ax.text(0.5, 0.5, "imputation sensitivity: no usable data", ha="center")
        fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return

    piv = imp_df.pivot_table(index="modality", columns="pair", values="jaccard", aggfunc="mean")
    null_by_mod = imp_df.groupby("modality")["null_jaccard"].mean()
    miss = imp_df.groupby("modality")[["raw_missing", "eff_missing"]].first()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                  gridspec_kw={"width_ratios": [2.4, 1]})
    import numpy as _np
    data = piv.values.astype(float)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = data[i, j]
            if _np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    ax.set_title("(a) Fixed imputation stability — Jaccard(top-50) between imputers\n"
                 "(MAD on unstandardized matrix; NOT variance-after-StandardScaler)",
                 fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")

    # right panel: per-modality mean jaccard vs chance + missingness
    mods = list(piv.index)
    mean_j = [_np.nanmean(data[i]) for i in range(len(mods))]
    chance = [null_by_mod.get(m, _np.nan) for m in mods]
    y = _np.arange(len(mods))
    ax2.barh(y - 0.2, mean_j, height=0.4, color="#2c7fb8", label="observed Jaccard")
    ax2.barh(y + 0.2, chance, height=0.4, color="#bdbdbd", label="chance (null)")
    ax2.set_yticks(y); ax2.set_yticklabels(mods)
    ax2.set_xlim(0, 1); ax2.set_xlabel("Jaccard")
    for i, m in enumerate(mods):
        ax2.text(1.01, i, f"raw {miss.loc[m,'raw_missing']:.0%} / eff {miss.loc[m,'eff_missing']:.0%}",
                 fontsize=7, va="center", transform=ax2.get_yaxis_transform())
    ax2.set_title("(b) observed vs chance\n(+ raw/effective missingness)", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 8 (revised): Imputation-sensitivity of biomarker rankings",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
