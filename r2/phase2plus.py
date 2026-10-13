"""
PHASE 2-4 + 7 — analyses that reuse the frozen best-fold model/embeddings.

  Phase 2 (R1.2, R2): cluster_validity_report + composition heatmap + technical-
                      confounder (dataset source) cross-tab.
  Phase 3 (R1.1):     stratified_survival_analysis (per-cancer KM/log-rank + BH-FDR).
  Phase 4 (R1.4):     attention-vs-reconstruction biomarker overlap (real numbers) +
                      curated-gene benchmark IF COSMIC/MSigDB files are present, else
                      NOT RUN with exact instructions.
  Phase 7:            row-alignment / NaN / finite consistency checks.

All numbers come from real runs. Non-significant / low values are reported as-is.
"""
import os, sys, json, time, pickle, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import torch
import nbdefs as nb
from nbdefs import config, device, SEED, HiOmicsFormer, MultiOmicsDataset, AttentionBiomarkerExtractor
from torch.utils.data import DataLoader
from hiomicsformer_revision_addons import (
    cluster_validity_report, benchmark_biomarkers_vs_curated)
from hiomicsformer_revision_survival_ext import stratified_survival_analysis

CACHE = os.path.join(HERE, "cache")
RES = config.results_path
FIG = config.figures_path
CURATED_DIR = os.path.join(ROOT, "data", "curated")
notes = {}


def log(m): print(m, flush=True)


def load_all():
    d = np.load(os.path.join(CACHE, "processed_data.npz"))
    processed = {m: d[m] for m in config.modalities}
    with open(os.path.join(CACHE, "meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    config.num_clusters = meta["num_clusters"]
    with open(os.path.join(CACHE, "preprocessor.pkl"), "rb") as fh:
        preprocessor = pickle.load(fh)
    emb = np.load(os.path.join(RES, "best_fold_embeddings.npy"))
    clus = np.load(os.path.join(RES, "best_fold_clusters.npy"))
    return processed, meta, preprocessor, emb, clus


# ============================================================================
def phase2(meta, clusters):
    log("\n########## PHASE 2 — CLUSTER VALIDITY ##########")
    cancer_labels = np.asarray(meta["cancer_labels"])
    cv = cluster_validity_report(clusters, cancer_labels,
                                 save_prefix=os.path.join(RES, "cluster_validity"))
    # composition heatmap (row-normalized cluster x cancer type)
    rn = cv["composition_rownorm"]
    fig, ax = plt.subplots(figsize=(1.1 * rn.shape[1] + 2, 0.5 * rn.shape[0] + 2))
    im = ax.imshow(rn.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rn.shape[1])); ax.set_xticklabels(rn.columns, rotation=45, ha="right")
    ax.set_yticks(range(rn.shape[0])); ax.set_yticklabels([f"C{c}" for c in rn.index])
    for i in range(rn.shape[0]):
        for j in range(rn.shape[1]):
            v = rn.values[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("cancer type"); ax.set_ylabel("cluster")
    ax.set_title(f"Cluster composition (row-normalized)\n"
                 f"purity={cv['overall_purity']:.2f}  Cramér's V={cv['cramers_v']:.2f}  "
                 f"NMI={cv['nmi']:.2f}  ARI={cv['ari']:.2f}", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig11_cluster_composition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- technical-confounder check: dataset source (clustering vs classification) ----
    tech = None
    try:
        loader = nb.MLOmicsDataLoader(config.data_path, data_variant=config.data_variant)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            disc = loader.discover_cancer_types()
        src_by_cancer = {code: source for code, _, source in disc}
        source = np.array([src_by_cancer.get(c, "unknown") for c in cancer_labels])
        if len(set(source)) > 1:
            from scipy import stats as _st
            ct = pd.crosstab(pd.Series(clusters, name="cluster"),
                             pd.Series(source, name="source"))
            chi2, p, dof, _ = _st.chi2_contingency(ct.values)
            v = float(np.sqrt(chi2 / (len(clusters) * (min(ct.shape) - 1))))
            ct.to_csv(os.path.join(RES, "cluster_vs_technical_source.csv"))
            tech = {"variable": "dataset_source(clustering|classification)",
                    "cramers_v": v, "chi_p": float(p)}
            log(f"  technical-source cross-tab: Cramér's V={v:.3f}, p={p:.2e} "
                f"(clusters {'DO' if v > 0.15 else 'do NOT strongly'} track dataset source)")
    except Exception as e:
        tech = {"error": str(e)}

    out = {"overall_purity": float(cv["overall_purity"]), "cramers_v": float(cv["cramers_v"]),
           "chi_p": float(cv["chi_p"]), "nmi": float(cv["nmi"]), "ari": float(cv["ari"]),
           "n_sig_enrichments_fdr05": int(len(cv["enrichment_sig"])),
           "technical_source": tech}
    with open(os.path.join(RES, "cluster_validity_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    notes["phase2"] = out
    return out


# ============================================================================
def phase3(meta, emb):
    log("\n########## PHASE 3 — STRATIFIED SURVIVAL ##########")
    survival_data = meta["survival_data"]
    if survival_data is None:
        notes["phase3"] = {"status": "NOT RUN — no survival data"}
        log("  NOT RUN — survival_data is None"); return
    sd = survival_data.copy()
    if sd.index.duplicated().any():
        log(f"  de-duplicating {int(sd.index.duplicated().sum())} repeated survival barcodes (keep first)")
        sd = sd[~sd.index.duplicated(keep="first")]
    res = stratified_survival_analysis(
        embeddings=emb, cancer_labels=np.asarray(meta["cancer_labels"]),
        survival_df=sd, sample_ids=meta["sample_ids"],
        save_path=os.path.join(FIG, "fig7b_stratified_survival.png"), seed=SEED)
    res.to_csv(os.path.join(RES, "stratified_survival.csv"), index=False)
    tested = int(res["logrank_p"].notna().sum())
    n_sig = int((res.get("logrank_fdr", pd.Series(dtype=float)) < 0.05).sum())
    notes["phase3"] = {"types_tested": tested, "n_sig_fdr05": n_sig,
                       "min_logrank_p": float(np.nanmin(res["logrank_p"].values)) if tested else None}
    log(f"  cancer types tested={tested}, significant within-cancer (FDR<0.05)={n_sig}")


# ============================================================================
def phase4(processed, meta, preprocessor):
    log("\n########## PHASE 4 — BIOMARKER OVERLAP + CURATED BENCHMARK ##########")
    feature_dims = {m: processed[m].shape[1] for m in config.modalities}
    full_dataset = MultiOmicsDataset(processed, meta["labels_encoded"],
                                     survival_data=None, sample_ids=meta["sample_ids"])
    model = HiOmicsFormer(feature_dims, config).to(device)
    model.load_state_dict(torch.load(os.path.join(CACHE, "best_model.pt"), map_location=device))
    model.eval()
    loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    ext = AttentionBiomarkerExtractor(model, preprocessor, device)
    attn = ext.extract_attention_biomarkers(loader, top_k=config.top_k_biomarkers)
    recon = ext.extract_reconstruction_biomarkers(loader, top_k=config.top_k_biomarkers)

    overlap_rows, consensus = [], {}
    for mod in config.modalities:
        a = set(b["name"] for b in attn.get(mod, []))
        r = set(b["name"] for b in recon.get(mod, []))
        inter = a & r; union = a | r
        jac = len(inter) / len(union) if union else 0.0
        overlap_rows.append({"modality": mod, "attn_n": len(a), "recon_n": len(r),
                             "overlap": len(inter), "jaccard": round(jac, 3),
                             "overlap_pct_of_50": round(100 * len(inter) / max(len(a), 1), 1)})
        consensus[mod] = sorted(inter)
        log(f"  {mod:6s}: attn∩recon={len(inter)}/50  Jaccard={jac:.3f}  consensus_genes={len(inter)}")
    pd.DataFrame(overlap_rows).to_csv(os.path.join(RES, "biomarker_method_overlap.csv"), index=False)
    # save discovered gene lists (real names)
    with open(os.path.join(RES, "biomarkers_attention.json"), "w") as fh:
        json.dump({m: [b["name"] for b in attn.get(m, [])] for m in config.modalities}, fh, indent=2)
    with open(os.path.join(RES, "biomarkers_consensus.json"), "w") as fh:
        json.dump(consensus, fh, indent=2)

    # ---- curated benchmark (only if files present) ----
    curated_status = {}
    for mod in config.modalities:
        rep = meta["feature_names"].get(mod, [])
        # only gene-symbol modalities are meaningful for COSMIC/MSigDB (mRNA/Methy/CNV are gene-level; miRNA is not)
    sets = _load_curated_sets()
    if not sets:
        msg = ("NOT RUN — curated gene-set files absent. To enable, place TAB/GMT files under "
               f"{CURATED_DIR}/ :\n"
               "  * COSMIC Cancer Gene Census: cosmic_cgc.tsv (col 'Gene Symbol'); "
               "register free at https://cancer.sanger.ac.uk/census then download.\n"
               "  * MSigDB Hallmark: h.all.v2024.1.Hs.symbols.gmt ; "
               "https://www.gsea-msigdb.org/gsea/msigdb (login/email required).\n"
               "  * MSigDB C6 oncogenic: c6.all.v2024.1.Hs.symbols.gmt (same site).")
        log("  " + msg.replace("\n", "\n  "))
        curated_status = {"status": "NOT RUN — curated files absent", "expected_dir": CURATED_DIR}
    else:
        for mod in ["mRNA", "Methy", "CNV"]:
            bg = list(preprocessor.get_feature_names(mod))
            disc = consensus.get(mod) or [b["name"] for b in attn.get(mod, [])]
            bench = benchmark_biomarkers_vs_curated(disc, bg, sets, label=mod)
            bench.to_csv(os.path.join(RES, f"curated_benchmark_{mod}.csv"), index=False)
        curated_status = {"status": "RAN", "sets": list(sets.keys())}
    notes["phase4"] = {"method_overlap": overlap_rows, "curated": curated_status}


def _load_curated_sets():
    sets = {}
    if not os.path.isdir(CURATED_DIR):
        return sets
    # COSMIC CGC tsv
    for f in glob.glob(os.path.join(CURATED_DIR, "*cosmic*csv")) + glob.glob(os.path.join(CURATED_DIR, "*cosmic*tsv")) + glob.glob(os.path.join(CURATED_DIR, "*census*csv")) + glob.glob(os.path.join(CURATED_DIR, "*census*tsv")):
        try:
            sep = "\t" if f.endswith("tsv") else ","
            df = pd.read_csv(f, sep=sep)
            col = next((c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()), df.columns[0])
            sets["COSMIC_CGC"] = set(df[col].astype(str))
        except Exception:
            pass
    # GMT files (MSigDB)
    for f in glob.glob(os.path.join(CURATED_DIR, "*.gmt")):
        name = os.path.basename(f).split(".")[0]
        genes = set()
        for line in open(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                genes.update(parts[2:])
        if genes:
            sets[name] = genes
    return sets


# ============================================================================
def phase7(processed, meta, emb, clusters):
    log("\n########## PHASE 7 — CONSISTENCY CHECKS ##########")
    checks = {}
    N = len(meta["cancer_labels"])
    checks["row_alignment"] = bool(len(emb) == N == len(meta["sample_ids"]) == len(clusters))
    checks["embeddings_finite"] = bool(np.isfinite(emb).all())
    checks["processed_finite"] = bool(all(np.isfinite(processed[m]).all() for m in config.modalities))
    # deterministic inference: forward twice, compare
    feature_dims = {m: processed[m].shape[1] for m in config.modalities}
    fd = MultiOmicsDataset(processed, meta["labels_encoded"], sample_ids=meta["sample_ids"])
    model = HiOmicsFormer(feature_dims, config).to(device)
    model.load_state_dict(torch.load(os.path.join(CACHE, "best_model.pt"), map_location=device))
    model.eval()

    def fwd():
        ld = DataLoader(fd, batch_size=config.batch_size, shuffle=False)
        outs = []
        with torch.no_grad():
            for b in ld:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                outs.append(model(b)["z"].cpu().numpy())
        return np.concatenate(outs)
    e1, e2 = fwd(), fwd()
    checks["deterministic_inference_maxdiff"] = float(np.abs(e1 - e2).max())
    checks["deterministic_inference_ok"] = bool(np.allclose(e1, e2, atol=1e-5))
    for k, v in checks.items():
        log(f"  {k}: {v}")
    if not (checks["row_alignment"] and checks["embeddings_finite"] and checks["processed_finite"]):
        raise SystemExit("CONSISTENCY FAILURE — see checks above")
    notes["phase7"] = checks
    with open(os.path.join(RES, "consistency_checks.json"), "w") as fh:
        json.dump(checks, fh, indent=2)


def main():
    t0 = time.time()
    processed, meta, preprocessor, emb, clusters = load_all()
    phase2(meta, clusters)
    phase3(meta, emb)
    phase4(processed, meta, preprocessor)
    phase7(processed, meta, emb, clusters)
    with open(os.path.join(RES, "phase2plus_notes.json"), "w") as fh:
        json.dump(notes, fh, indent=2, default=str)
    log(f"\nPHASE 2-4+7 COMPLETE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
