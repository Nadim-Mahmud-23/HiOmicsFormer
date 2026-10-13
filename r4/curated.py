"""
E5 - Curated cancer-gene benchmarking of the model-derived feature scores  [Reviewer 1.4].

Scores are recomputed with the frozen models exactly as in R2 (AttentionBiomarkerExtractor):
  * 'gradient' score  = mean |d H(q) / d x_j|, H(q) the entropy of the DEC soft assignment
                        (the R2 code labels this 'attention_gradient'; it contains no attention term);
  * 'reconstruction' score = mean(err_j) * var(err_j) of the squared reconstruction error.
Reference sets (gene symbols):
  COSMIC Cancer Gene Census (v99 membership, as recorded in the OncoKB cancer gene list),
  OncoKB cancer gene list (all entries), MSigDB Hallmark (union of 50 sets) and MSigDB C6
  oncogenic signatures (union of 189 sets; v2023.2.Hs).
Background = the modality's 2,000-feature MAD panel (the features that could be scored), never the
genome. Two tests per modality x score x reference: (i) hypergeometric over-representation of the
top-50 list; (ii) a rank test over all 2,000 scores (one-sided Mann-Whitney; AUC = probability that
a curated gene outranks a non-curated one). Benjamini-Hochberg across all tests of a model.
Symbols that are purely numeric (spreadsheet date conversions, e.g. '43899') are counted and
reported; they cannot be matched to any curated set.
Output: results_revision/r4/curated_benchmark.csv, curated_benchmark_summary.json, feature_scores_<model>.csv
"""
import os, re
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from common import ROOT, OUT, EXT, config, log, save_json, load_cache, load_model, MultiOmicsDataset
from nbdefs import AttentionBiomarkerExtractor, device
from torch.utils.data import DataLoader

NUMERIC = re.compile(r"^\d+(\.\d+)?$")


def gmt_union(path):
    genes = set()
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        genes.update(g.strip().upper() for g in parts[2:] if g.strip())
    return genes


def reference_sets():
    ok = pd.read_csv(os.path.join(EXT, "oncokb_cancerGeneList.txt"), sep="\t")
    sym = ok["Hugo Symbol"].astype(str).str.upper()
    return {"COSMIC CGC (v99)": set(sym[ok["COSMIC CGC (v99)"] == "Yes"]),
            "OncoKB cancer genes": set(sym),
            "MSigDB Hallmark": gmt_union(os.path.join(ROOT, "data", "curated", "hallmark.gmt")),
            "MSigDB C6 oncogenic": gmt_union(os.path.join(ROOT, "data", "curated", "c6_oncogenic.gmt"))}


def scores(model_path, cohort):
    processed, meta, pre = load_cache(cohort)
    fd = {m: processed[m].shape[1] for m in config.modalities}
    model = load_model(model_path, fd, meta["num_clusters"])
    ds = MultiOmicsDataset(processed, meta["labels_encoded"], sample_ids=meta["sample_ids"])
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    ext = AttentionBiomarkerExtractor(model, pre, device)
    grad = ext._gradient_feature_importance(loader, top_k=10 ** 6)
    rec = ext.extract_reconstruction_biomarkers(loader, top_k=10 ** 6)
    out = {}
    for m in config.modalities:
        g = pd.Series({b["name"]: b["importance"] for b in grad[m]})
        r = pd.Series({b["name"]: b["importance"] for b in rec[m]})
        out[m] = pd.DataFrame({"gradient": g, "reconstruction": r})
    return out


def main():
    refs = reference_sets(); rows = []; summary = {}
    for model_name, cohort, path in (
            ("full-cohort model", "all", "results_revision/all_uncorrected/best_model.pt"),
            ("single-archive model", "single_archive", "results_revision/single_archive/best_model.pt")):
        sc = scores(os.path.join(ROOT, path), cohort)
        frames = []
        for m, df in sc.items():
            d = df.copy(); d["modality"] = m; frames.append(d.rename_axis("feature").reset_index())
        pd.concat(frames).to_csv(os.path.join(OUT, f"feature_scores_{cohort}.csv"), index=False)
        summary[model_name] = {}
        for m in ("mRNA", "Methy", "CNV"):
            df = sc[m].copy()
            df.index = [str(i).replace(".", "-").upper() for i in df.index]
            n_numeric = int(sum(bool(NUMERIC.match(i)) for i in df.index))
            top = {s: set(df[s].sort_values(ascending=False).index[:50]) for s in ("gradient", "reconstruction")}
            top["consensus"] = top["gradient"] & top["reconstruction"]
            summary[model_name][m] = {"panel_size": len(df), "numeric_symbols_in_panel": n_numeric,
                                      "numeric_symbols_in_top50_gradient": int(sum(bool(NUMERIC.match(i)) for i in top["gradient"])),
                                      "consensus_size": len(top["consensus"]),
                                      "jaccard_gradient_vs_reconstruction": len(top["consensus"]) / len(top["gradient"] | top["reconstruction"]),
                                      "top10_gradient": list(df["gradient"].sort_values(ascending=False).index[:10])}
            N = len(df)
            for ref_name, ref in refs.items():
                in_ref = df.index.isin(list(ref)); K = int(in_ref.sum())
                for s, members in top.items():
                    n = len(members); x = int(sum(g in ref for g in members))
                    p = float(hypergeom.sf(x - 1, N, K, n)) if n else float("nan")
                    rows.append({"model": model_name, "modality": m, "score": s, "reference": ref_name, "test": "top-50 hypergeometric",
                                 "panel_N": N, "panel_in_reference": K, "list_n": n, "overlap": x,
                                 "expected": n * K / N if N else float("nan"),
                                 "fold": (x / (n * K / N)) if n and K else float("nan"), "p": p, "auc": float("nan")})
                for s in ("gradient", "reconstruction"):
                    a, b = df[s].values[in_ref], df[s].values[~in_ref]
                    if len(a) and len(b):
                        u = mannwhitneyu(a, b, alternative="greater")
                        rows.append({"model": model_name, "modality": m, "score": s, "reference": ref_name,
                                     "test": "rank (Mann-Whitney, all 2,000)", "panel_N": N, "panel_in_reference": K,
                                     "list_n": N, "overlap": K, "expected": float("nan"), "fold": float("nan"),
                                     "p": float(u.pvalue), "auc": float(u.statistic / (len(a) * len(b)))})
    res = pd.DataFrame(rows)
    for mn in res["model"].unique():
        sel = (res["model"] == mn) & res["p"].notna()
        res.loc[sel, "fdr"] = multipletests(res.loc[sel, "p"], method="fdr_bh")[1]
        summary[mn]["n_tests"] = int(sel.sum())
        summary[mn]["n_fdr_below_0.05"] = int((res.loc[sel, "fdr"] < 0.05).sum())
        summary[mn]["min_fdr"] = float(res.loc[sel, "fdr"].min())
    res.to_csv(os.path.join(OUT, "curated_benchmark.csv"), index=False)
    log(res[res["fdr"] < 0.25].round(4).to_string(index=False) if (res["fdr"] < 0.25).any() else "no test with FDR < 0.25")
    save_json(summary, "curated_benchmark_summary.json")


if __name__ == "__main__":
    main()
