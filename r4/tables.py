"""LaTeX tables for the manuscript and supplement, generated directly from results files
(no hand-typed numbers). Output: HiOmicsFormer_Elsevier_Submission/tables/*.tex"""
import os, json
import numpy as np
import pandas as pd
from common import ROOT, OUT, config, load_cache

TAB = os.path.join(ROOT, "HiOmicsFormer_Elsevier_Submission", "tables")
RES = os.path.join(ROOT, "results_revision")
os.makedirs(TAB, exist_ok=True)


def J(p):
    return json.load(open(p))


LONG_CAPTIONS = {'tabS_known_subtypes.tex': "Agreement of within-type K-means partitions ($K$ = number of known classes) with PanCanAtlas molecular subtypes and immune subtypes, for the model's embedding and for the first 50 principal components of the input.", 'tabS_curated.tex': "Curated gene-set benchmarking of the feature scores (gene-level modalities). Top-50: hypergeometric test of the top-50 list (overlap/list size, expected overlap). Rank: one-sided Mann--Whitney test over all 2,000 scores (AUC = probability that a curated gene outranks a non-curated one). Background: the modality's 2,000-feature panel. FDR: Benjamini--Hochberg within model.", 'tabS_audit.tex': "Scale of every MLOmics ``Original'' file in the analysed release, computed from the raw CSV files (features $\\times$ patients). ``z-scored in file'': median absolute feature mean $<0.05$ and median feature SD within $0.05$ of 1. ``constant'': fraction of features with zero variance."}


def w(name, s):
    if name in LONG_CAPTIONS:   # caption inside the longtable (first row)
        head, rest = s.split("\n", 1)
        s = head + "\n\\caption{" + LONG_CAPTIONS[name] + "}\\\\\n" + rest
    open(os.path.join(TAB, name), "w").write(s); print("wrote", name)


def f3(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def t_baselines():
    a = pd.read_csv(os.path.join(OUT, "baselines_all.csv")).set_index("method")
    s = pd.read_csv(os.path.join(OUT, "baselines_single_archive.csv")).set_index("method")
    order = ["HiOmicsFormer"] + [m for m in s.sort_values("nmi_mean", ascending=False).index if m != "HiOmicsFormer"]
    lines = []
    for m in order:
        r1, r2 = a.loc[m], s.loc[m]
        name = f"\\textbf{{{m}}}" if m == "HiOmicsFormer" else m
        lines.append(f"{name} & {r1.silhouette_mean:.3f} & {r1.nmi_mean:.3f} $\\pm$ {r1.nmi_sd:.3f} & {r1.ari_mean:.3f} & "
                     f"{r2.silhouette_mean:.3f} & {r2.nmi_mean:.3f} $\\pm$ {r2.nmi_sd:.3f} & {r2.ari_mean:.3f} \\\\")
    w("tab_baselines.tex", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}\n\\toprule\n"
      " & \\multicolumn{3}{c}{Full cohort (11 types, $N=3{,}044$)} & \\multicolumn{3}{c}{Single archive (8 types, $N=2{,}269$)} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nMethod & Silhouette$^{a}$ & NMI & ARI & Silhouette$^{a}$ & NMI & ARI \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n")


def t_sensitivity():
    rows = []
    for coh, lab in (("all", "11"), ("all_fixed", "13$^{b}$")):
        p = os.path.join(OUT, f"sensitivity_{coh}.json")
        if not os.path.exists(p):
            continue
        for v, r in J(p).items():
            nice = {"reference": "Published pipeline", "per_type_standardized": "Files standardised within type",
                    "no_miRNA": "Without miRNA", "mean_imputation": "Mean imputation",
                    "complete_features": "Fully observed features"}[v]
            il = r["input_level"]
            ain = il.get("input_pc1_auc_scale_group", il.get("input_pc1_auc_archive"))   # GBM/LGG/OV vs rest
            amd = r.get("pc1_auc_scale_group", r.get("pc1_auc_archive"))
            rows.append(f"{nice} [{lab}] & {f3(ain)} & {f3(il['kmeans_nmi_type'])} & "
                        f"{r['pc1_variance_pct']:.1f} & {f3(amd)} & {f3(r['test_fold']['nmi'])} & "
                        f"{', '.join(str(x) for x in r['cluster_sizes'][:3])}{', \\ldots' if len(r['cluster_sizes']) > 3 else ''} \\\\")
    w("tab_sensitivity.tex", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccl}\n\\toprule\n"
      " & \\multicolumn{2}{c}{Input (no model)} & \\multicolumn{4}{c}{HiOmicsFormer} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
      "Preprocessing & PC1 AUC & NMI & PC1 \\% & PC1 AUC & NMI & Largest clusters \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")


def t_known_subtypes():
    k = pd.read_csv(os.path.join(OUT, "known_subtypes.csv"))
    lines = []
    for _, r in k.sort_values(["reference", "model", "cancer_type"]).iterrows():
        lines.append(f"{r.reference} & {r.model} & {r.cancer_type} & {r.n_labelled} & {r.k_known} & "
                     f"{r.ARI_model:.3f} & {r.ARI_input_PCA:.3f} & {r.NMI_model:.3f} & {r.NMI_input_PCA:.3f} \\\\")
    w("tabS_known_subtypes.tex", "\\begin{longtable}{lllrrcccc}\n\\toprule\nReference & Embedding & Type & $n$ & $K$ & ARI model & ARI input & NMI model & NMI input \\\\\n\\midrule\n\\endhead\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")


def t_cindex():
    c = pd.read_csv(os.path.join(OUT, "survival_cindex.csv"))
    lines = [f"{r.model} & {r.cancer_type} & {r.n} & {r.events} & {f3(r.cindex_model)} & {f3(r.cindex_input)} \\\\" for _, r in c.iterrows()]
    w("tabS_cindex.tex", "\\begin{tabular}{llrrcc}\n\\toprule\nEmbedding & Type & $n$ & Deaths & C-index (model PCs) & C-index (input PCs) \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def t_stratified_survival():
    lines = []
    for lab, d in (("Full cohort", "all_uncorrected"), ("Single archive", "single_archive")):
        s = pd.read_csv(os.path.join(RES, d, "stratified_survival.csv"))
        for _, r in s.iterrows():
            p = r.get("logrank_p", np.nan); q = r.get("logrank_fdr", np.nan)
            sizes = f"{int(r['n_sub0'])}/{int(r['n_sub1'])}" if pd.notna(r.get('n_sub0')) and pd.notna(r.get('n_sub1')) else "--"
            lines.append(f"{lab} & {r['cancer_type']} & {int(r['n'])} & {sizes} & {f3(p)} & {f3(q)} & {str(r.get('note', '')).replace('subtype too small', 'one cluster too small').replace('too few survival samples', 'no survival data').replace('ok', '')} \\\\")
    w("tabS_stratified_survival.tex", "\\begin{tabular}{llrcccl}\n\\toprule\nCohort & Type & $n$ & Cluster sizes & log-rank $p$ & BH-FDR & Note \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def t_curated():
    c = pd.read_csv(os.path.join(OUT, "curated_benchmark.csv"))
    lines = []
    for _, r in c.iterrows():
        if r.test.startswith("top"):
            stat = f"{int(r.overlap)}/{int(r.list_n)} (exp.\\ {r.expected:.1f})"
        else:
            stat = f"AUC {r.auc:.3f}"
        lines.append(f"{r.model.replace(' model','')} & {r.modality} & {r.score} & {r.reference} & {stat} & {r.p:.3g} & {r.fdr:.3g} \\\\")
    w("tabS_curated.tex", "\\begin{longtable}{lllllcc}\n\\toprule\nModel & Modality & Score & Reference & Result & $p$ & FDR \\\\\n\\midrule\n\\endhead\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")


def t_audit():
    a = pd.read_csv(os.path.join(OUT, "normalization_audit.csv"))
    lines = [f"{r.folder} & {r.modality} & {r.features} & {r.samples} & {r.median_abs_feature_mean:.3f} & {r.median_feature_sd:.3f} & "
             f"{r.frac_constant_features:.2f} & {r['min']:.2f} to {r['max']:.2f} & {'yes' if r.zscored_within_file else '\\textbf{no}'} \\\\"
             for _, r in a.iterrows()]
    w("tabS_audit.tex", "\\begin{longtable}{llrrccccc}\n\\toprule\nFolder & Modality & Features & Patients & median $|$mean$|$ & median SD & constant & range & z-scored in file \\\\\n\\midrule\n\\endhead\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")


def t_perfold():
    lines = []
    for lab, d in (("Full cohort", "all_uncorrected"), ("Single archive", "single_archive")):
        cv = J(os.path.join(RES, d, "cv_metrics.json"))["cv_summary"]
        for met in ("silhouette", "nmi", "ari"):
            v = cv[met]["values"]
            lines.append(f"{lab} & {met.upper() if met != 'silhouette' else 'Silhouette'} & " + " & ".join(f"{x:.3f}" for x in v)
                         + f" & {cv[met]['mean']:.3f} $\\pm$ {cv[met]['std']:.3f} \\\\")
    w("tabS_perfold.tex", "\\begin{tabular}{llcccccc}\n\\toprule\nCohort & Metric & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean $\\pm$ SD \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def t_sweep():
    s = pd.read_csv(os.path.join(RES, "dec_collapse_sweep.csv"))
    lines = [f"{int(r.Kc)} & {r.lambda_DEC} & {r.test_silhouette:.3f} & {r.test_nmi:.3f} & {r.test_ari:.3f} & {int(r.effective_clusters)} \\\\" for _, r in s.iterrows()]
    w("tabS_sweep.tex", "\\begin{tabular}{rrcccr}\n\\toprule\n$K_c$ & $\\lambda_{\\text{DEC}}$ & Silhouette & NMI & ARI & Clusters used \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def t_composition():
    _, meta, _ = load_cache("all")
    lab = np.asarray([str(x) for x in meta["cancer_labels"]])
    clu = np.load(os.path.join(RES, "all_uncorrected", "clusters.npy"))
    tab = pd.crosstab(pd.Series(clu, name="cluster"), pd.Series(lab, name="type"))
    tab = tab.loc[tab.sum(1).sort_values(ascending=False).index]
    head = " & ".join(tab.columns)
    lines = [f"C{c} & " + " & ".join(str(v) for v in tab.loc[c].values) + f" & {tab.loc[c].sum()} \\\\" for c in tab.index]
    w("tabS_composition.tex", f"\\begin{{tabular}}{{l{'r' * tab.shape[1]}r}}\n\\toprule\nCluster & {head} & Total \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def t_cleaning():
    r = J(os.path.join(RES, "cleaning_report.json"))["per_file"]
    rows = [(k.split("/")[0], k.split("/")[1], v) for k, v in r.items() if k.split("/")[0] not in ("LUAD", "BRCA")]
    lines = [f"GS-{t} & {m} & {v['before']:,} & {v['after']:,} & {v['dropped']:,} \\\\" for t, m, v in rows]
    tot = sum(v["dropped"] for _, _, v in rows)
    w("tabS_cleaning.tex", "\\begin{tabular}{llrrr}\n\\toprule\nFile & Modality & Columns before & After & Removed \\\\\n\\midrule\n"
      + "\n".join(lines) + f"\n\\midrule\nTotal & & & & {tot:,} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def t_duplicate_identity():
    d = J(os.path.join(OUT, "duplicate_coad_identity.json"))
    lines = [f"{m} & {v['shared_features']:,} & {v['shared_patients']} & {v['per_feature_corr_min']:.6f} & "
             f"{v['restandardised_abs_diff_median']:.1e} & {v['restandardised_abs_diff_max']:.1e} \\\\" for m, v in d.items()]
    w("tabS_duplicate_identity.tex", "\\begin{tabular}{lrrccc}\n\\toprule\nModality & Shared features & Shared patients & min.\\ per-feature $r$ & "
      "median $|\\Delta|$ & max.\\ $|\\Delta|$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    for f in (t_cleaning, t_duplicate_identity, t_baselines, t_sensitivity, t_known_subtypes, t_cindex, t_stratified_survival, t_curated, t_audit,
              t_perfold, t_sweep, t_composition):
        try:
            f()
        except (FileNotFoundError, KeyError) as e:
            print(f"skip {f.__name__}: {e!r}")
