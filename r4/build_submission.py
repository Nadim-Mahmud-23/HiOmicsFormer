"""
Assemble the revised manuscript and letters, filling every reported number from results files.

  1. compute_numbers(): every number quoted in the text, read from results_revision/ artifacts;
  2. manuscript: preamble + parts (intro/methods, body) -> placeholders \\PH{KEY} replaced;
     bibliography = only cited keys, in order of first citation (elsarticle-num numbering);
  3. letters: {{KEY}} placeholders replaced.
Any unfilled placeholder or missing bibitem is a hard error.
Usage: python3 r4/build_submission.py [--pages N --supp-pages M]
"""
import os, re, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "HiOmicsFormer_Elsevier_Submission")
PARTS = os.path.join(HERE, "manuscript_parts")
RES = os.path.join(ROOT, "results_revision"); R4 = os.path.join(RES, "r4")
LET = os.path.join(ROOT, "submission")


def J(p):
    return json.load(open(p))


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def compute_numbers():
    N = {}
    sens = J(os.path.join(R4, "sensitivity_all.json"))
    ref, std = sens["reference"], sens["per_type_standardized"]
    N["AUCIN"] = f"{ref['input_level']['input_pc1_auc_archive']:.3f}"
    N["AUCINSTD"] = f"{std['input_level']['input_pc1_auc_archive']:.3f}"
    N["AUCMSTD"] = f"{std['pc1_auc_archive']:.3f}"
    N["KMSTD"] = f"{std['input_level']['kmeans_nmi_type']:.3f}"
    N["KMREF"] = f"{ref['input_level']['kmeans_nmi_type']:.3f}"
    N["NMIREFTEST"] = f"{ref['test_fold']['nmi']:.3f}"; N["NMISTDTEST"] = f"{std['test_fold']['nmi']:.3f}"
    N["UARCHSTD"] = pct(std["association_archive"]["uncertainty_coefficient"])
    f13 = J(os.path.join(R4, "sensitivity_all_fixed.json"))["reference"]
    N["U13SCALE"] = pct(f13["association_scale_group"]["uncertainty_coefficient"])
    N["U13ARCH"] = pct(f13["association_archive"]["uncertainty_coefficient"])
    N["AUC13SCALE"] = f"{f13['pc1_auc_scale_group']:.3f}"
    from sklearn.decomposition import PCA
    zz = np.load(os.path.join(R4, "sensitivity_all_fixed_reference_embedding.npz"), allow_pickle=True)
    pc = PCA(1).fit_transform(zz["z"])[:, 0]; lb = zz["labels"]
    grp = np.isin(lb, ["GBM", "LGG", "OV"])
    if np.median(pc[grp]) < np.median(pc[~grp]):
        pc = -pc
    N["BRCAPC1"] = f"${np.median(pc[lb == 'BRCA']):.2f}$"
    N["SCALEPC1"] = f"{np.median(pc[grp]):.2f}"
    N["CLUPC1"] = f"${np.median(pc[~grp & (lb != 'BRCA')]):.2f}$"
    others = [v for k, v in sens.items() if k in ("no_miRNA", "mean_imputation", "complete_features")]
    aucs = [v["pc1_auc_archive"] for v in others]
    names = {"no_miRNA": "without miRNA", "mean_imputation": "with mean imputation", "complete_features": "on fully observed features"}
    parts = [f"{names[k]} {sens[k]['pc1_auc_archive']:.3f}" for k in ("no_miRNA", "mean_imputation", "complete_features") if k in sens]
    if min(aucs) > 0.95:
        N["SENSRESULT"] = ("left the axis in place (model AUC " + "; ".join(parts) + ")")
        N["SENSRESULTLETTER"] = ("In every case the dominant axis remained (AUC for separating GBM/LGG/OV: " + "; ".join(parts) + ").")
    else:
        N["SENSRESULT"] = ("changed the axis (model AUC " + "; ".join(parts) + ")")
        N["SENSRESULTLETTER"] = ("The axis changed in at least one variant (AUC: " + "; ".join(parts) + "); see Table 3.")
    bs = pd.read_csv(os.path.join(R4, "baselines_single_archive.csv")).set_index("method")
    ba = pd.read_csv(os.path.join(R4, "baselines_all.csv")).set_index("method")
    N["MAXNMI"] = f"{bs.drop('HiOmicsFormer')['nmi_mean'].max():.3f}"
    h = bs.loc["HiOmicsFormer"]; N["NMISINGLE"] = f"{h.nmi_mean:.3f} $\\pm$ {h.nmi_sd:.3f}"
    cvf = J(os.path.join(RES, "all_uncorrected", "cv_metrics.json"))["cv_summary"]
    N["SILFULL"] = f"{cvf['silhouette']['mean']:.3f} $\\pm$ {cvf['silhouette']['std']:.3f}"
    N["NMIFULL"] = f"{cvf['nmi']['mean']:.3f} $\\pm$ {cvf['nmi']['std']:.3f}"
    N["NMIFULLPLAIN"] = f"{cvf['nmi']['mean']:.3f} ± {cvf['nmi']['std']:.3f}"
    dg = J(os.path.join(RES, "all_uncorrected", "diagnostics.json"))
    N["PC1FULL"] = f"{dg['pc1_var_pct']:.1f}"
    sizes = sorted([s for s in dg["cluster_sizes"] if s > 0], reverse=True)
    N["CL1"] = f"{sizes[0]:,}"; N["CL2"] = f"{sizes[1]:,}"
    # composition of the second cluster
    import pickle, sys
    sys.path.insert(0, HERE)
    meta = pickle.load(open(os.path.join(ROOT, "r3", "cache_all", "meta.pkl"), "rb"))
    lab = np.asarray([str(x) for x in meta["cancer_labels"]]); clu = np.load(os.path.join(RES, "all_uncorrected", "clusters.npy"))
    c2 = pd.Series(clu).value_counts().index[1]
    in2 = lab[clu == c2]
    assert set(in2) <= {"GBM", "LGG", "OV"}, f"second cluster not purely GBM/LGG/OV: {set(in2)}"
    N["CL2OF"] = f"{len(in2):,}"
    asc = J(os.path.join(R4, "association.json"))["all"]
    N["SILCOMMONM"] = f"${asc['silhouette_model_partition_in_input_pca_space']:.3f}$"
    N["SILOWN"] = f"{asc['silhouette_model_partition_in_model_space']:.3f}"
    N["SILCOMMONK"] = f"{asc['silhouette_kmeans_partition_in_input_pca_space']:.3f}"
    N["UARCH"] = pct(asc["vs_archive"]["uncertainty_coefficient"])
    N["UTYPE"] = pct(asc["vs_cancer_type"]["uncertainty_coefficient"])
    ms = J(os.path.join(R4, "missingness.json"))
    N["AUCMODEL"] = f"{ms['all']['pc1_auc_for_archive']:.3f}"
    sp = ms["all"]["selected_panel"]
    N["MISSMIR"] = pct(sp["miRNA"]["missing_fraction"], 1); N["MISSMRNA"] = pct(sp["mRNA"]["missing_fraction"], 1)
    a0, a1 = sp["miRNA"]["missing_fraction_clustering_archive"], sp["miRNA"]["missing_fraction_classification_archive"]
    N["MISSWHERE"] = f"{pct(a1)}\\% of miRNA values in the classification-archive files and {pct(a0)}\\% in the clustering-archive files"
    N["MISSWHERELETTER"] = N["MISSWHERE"].replace("\\%", "%")
    N["MISSAUC"] = f"{ms['all']['missingness_auc_for_archive']['all_selected']:.2f}"
    N["MISSRHO"] = f"{ms['all']['pc_vs_missingness_spearman']['PC1']['all_selected']:.2f}"
    dp = J(os.path.join(R4, "duplicate_patient_test.json"))["frozen_model"]
    assert dp["classification_copy"]["frac_assigned_to_classification_cluster"] == 0.0
    N["DUPSHIFT"] = pct(abs(dp["paired_pc1_shift_as_fraction_of_archive_gap"]), 2)
    sa = J(os.path.join(RES, "single_archive", "diagnostics.json"))
    ss = sorted([s for s in sa["cluster_sizes"] if s > 0])
    N["SACL"] = f"{ss[0]:,} and {ss[1]:,} patients"
    ks = pd.read_csv(os.path.join(R4, "known_subtypes.csv"))
    sel = lambda ref: ks[(ks.reference == ref) & ((ks.model == "single-archive model") |
                         ((ks.model == "full-cohort model") & ks.cancer_type.isin(["GBM", "LGG", "OV"])))]
    p = sel("PanCanAtlas molecular subtype"); im = sel("Immune subtype (C1-C6)")
    N["ARIM"] = f"{p.ARI_model.median():.3f}"; N["ARII"] = f"{p.ARI_input_PCA.median():.3f}"
    N["NSUBTYPES"] = str(len(p))
    better = int((p.ARI_model > p.ARI_input_PCA).sum())
    N["SUBTYPEDETAIL"] = (f"The embedding agreed better than the input in {better} of {len(p)} cancer types; for the immune subtypes the "
                          f"medians were {im.ARI_model.median():.3f} and {im.ARI_input_PCA.median():.3f} "
                          f"({int((im.ARI_model > im.ARI_input_PCA).sum())} of {len(im)} types better).")
    ci = pd.read_csv(os.path.join(R4, "survival_cindex.csv"))
    sc = ci[(ci.model == "single-archive model") & ci.cindex_model.notna()]
    fc = ci[(ci.model == "full-cohort model") & ci.cindex_model.notna()]
    N["CINDEX"] = (f"Cross-validated Cox models on the leading components of the embedding reached median concordances of "
                   f"{sc.cindex_model.median():.2f} (single-archive model) and {fc.cindex_model.median():.2f} (full-cohort model) "
                   f"across the {len(sc)} types with at least ten deaths, similar to components of the input "
                   f"({sc.cindex_input.median():.2f} and {fc.cindex_input.median():.2f}; Supplementary Table~S10). "
                   f"Several types have few deaths (COAD: {int(ci[ci.cancer_type=='COAD'].events.iloc[0])}), so these estimates are imprecise.")
    N["CINDEXSENTENCE"] = (f"median concordance {sc.cindex_model.median():.2f} (single-archive model) and {fc.cindex_model.median():.2f} "
                           f"(full-cohort model) across {len(sc)} cancer types, similar to {sc.cindex_input.median():.2f} and "
                           f"{fc.cindex_input.median():.2f} for components of the input (Supplementary Table S10).")
    cs = J(os.path.join(R4, "curated_benchmark_summary.json"))
    cb = pd.read_csv(os.path.join(R4, "curated_benchmark.csv"))
    N["NCUR"] = str(int(cb.p.notna().sum())); N["NCURSIG"] = str(int((cb.fdr < 0.05).sum()))
    jac = max(v["jaccard_gradient_vs_reconstruction"] for m in cs.values() if isinstance(m, dict)
              for k, v in m.items() if isinstance(v, dict) and "jaccard_gradient_vs_reconstruction" in v)
    N["JACMAX"] = f"{jac:.2f}"
    best = cb.sort_values("fdr").iloc[0]
    N["CURDETAIL"] = ("every FDR was 1.00." if best.fdr > 0.995 else
                      f"the smallest FDR was {best.fdr:.2f} ({best.model}, {best.modality}, {best.reference}).")
    cv = J(os.path.join(RES, "all_uncorrected", "cluster_validity.json"))
    N["PURITY"] = f"{cv['purity']:.2f}"; N["NENRICH"] = str(cv["n_sig_type_enrichments_fdr05"])
    au = pd.read_csv(os.path.join(R4, "normalization_audit.csv"))
    zf = au.groupby("folder")["zscored_within_file"].all(); z = au[au.folder.isin(zf[zf].index)]
    assert len(zf[zf]) == 11, zf
    sg = J(os.path.join(R4, "scale_signature.json")); ss = sg["median_within_type_sd_summary"]
    N["SDMETHYG"], N["SDMETHYO"] = f"{ss['Methy']['GBM_LGG_OV']:.2f}", f"{ss['Methy']['other_types']:.2f}"
    N["SDCNVG"], N["SDCNVO"] = f"{ss['CNV']['GBM_LGG_OV']:.2f}", f"{ss['CNV']['other_types']:.2f}"
    N["SDMRNAG"], N["SDMRNAO"] = f"{ss['mRNA']['GBM_LGG_OV']:.2f}", f"{ss['mRNA']['other_types']:.2f}"
    N["PC1METHY"] = pct(sg["input_pc1_loading_share"]["Methy"])
    sa_ = sg["supervised_balanced_accuracy"]["single_archive"]
    N["RFSINGLE"] = f"{sa_['random forest (100 PCs, 300 trees)']:.2f}"; N["LRSINGLE"] = f"{sa_['logistic regression (100 PCs)']:.2f}"
    N["KNNSINGLE"] = f"{sa_['k-nearest neighbours (50 PCs, k=15)']:.2f}"
    sy = J(os.path.join(R4, "symbol_audit.json"))
    N["DATEMRNA"], N["DATEMETHY"], N["DATECNV"] = (str(sy[m]["universe_total_converted"]) for m in ("mRNA", "Methy", "CNV"))
    N["ZMEAN"] = f"{z.median_abs_feature_mean.max():.2f}"
    N["ZSD"] = f"{z.median_feature_sd.min():.2f}--{z.median_feature_sd.max():.2f}"
    return N


def bibliography(text, pool):
    keys = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    items = {}
    for m in re.finditer(r"(\\bibitem\[[^\]]*\]\{([^}]+)\}.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)", pool, re.S):
        items[m.group(2)] = m.group(1).strip()
    missing = [k for k in keys if k not in items]
    if missing:
        raise SystemExit(f"missing bibitems: {missing}")
    return "\\begin{thebibliography}{100}\n\n" + "\n\n".join(items[k] for k in keys) + "\n\n\\end{thebibliography}\n", len(keys)


PREAMBLE = r"""\documentclass[preprint,12pt,super]{elsarticle}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{lineno}
\modulolinenumbers[5]
\journal{Advances in Biomarker Sciences and Technology}
\begin{document}
"""


def text_counts():
    """Abstract words, body words (Introduction..Conclusion, excluding floats/equations), title words."""
    tex = open(os.path.join(SUB, "HiOmicsFormer_manuscript.tex")).read()
    strip = lambda t: re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?|[{}$~]", " ", t)
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    body = tex.split("\\section{Introduction}")[1].split("\\section*{Declaration of Competing Interest}")[0]
    body = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{(figure|table)\}", " ", body, flags=re.S)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.S)
    body = re.sub(r"%.*", " ", body)
    title = re.search(r"\\title\{(.*?)\}", tex).group(1)
    return {"ABSWORDS": str(len(strip(ab).split())), "WORDS": f"{len(strip(body).split()):,}",
            "WORDS_INT": len(strip(body).split()), "TITLEWORDS": str(len(title.split()))}


def pdf_pages(path):
    import subprocess
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def md_to_docx(md_path, docx_path):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)

    def add_runs(par, text):
        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                par.add_run(tok[2:-2]).bold = True
            elif tok.startswith("*") and len(tok) > 1:
                par.add_run(tok[1:-1]).italic = True
            elif tok.startswith("`"):
                r = par.add_run(tok[1:-1]); r.font.name = "Courier New"
            else:
                par.add_run(tok)

    lines = open(md_path).read().split("\n"); i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-| ]+\|$", lines[i + 1]):
            rows = [ln]; i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            t = doc.add_table(rows=len(cells), cols=len(cells[0])); t.style = "Table Grid"
            for r, row in enumerate(cells):
                for c, val in enumerate(row):
                    p = t.cell(r, c).paragraphs[0]; add_runs(p, val)
                    if r == 0:
                        for run in p.runs:
                            run.bold = True
            continue
        if ln.startswith("# "):
            doc.add_heading(ln[2:], level=1)
        elif ln.startswith("## "):
            doc.add_heading(ln[3:], level=2)
        elif ln.startswith("### "):
            doc.add_heading(ln[4:], level=3)
        elif ln.strip() == "---":
            pass
        elif re.match(r"^\s*- ", ln):
            add_runs(doc.add_paragraph(style="List Bullet"), re.sub(r"^\s*- ", "", ln))
        elif re.match(r"^\d+\. ", ln):
            add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", ln))
        elif ln.strip():
            add_runs(doc.add_paragraph(), ln)
        i += 1
    doc.save(docx_path)


def build_letters(N):
    N = dict(N)
    N.update({k: v for k, v in text_counts().items() if k != "WORDS_INT"})
    words = text_counts()["WORDS_INT"]
    N["WORDPCT"] = f"{100 * (7690 - words) / 7690:.0f}"
    N["PAGES"] = str(pdf_pages(os.path.join(SUB, "HiOmicsFormer_manuscript.pdf")))
    N["SUPPPAGES"] = str(pdf_pages(os.path.join(SUB, "Supplementary.pdf")))
    for name in ("response", "cover_letter"):
        t = open(os.path.join(LET, f"{name}_template.md")).read()
        for k, v in N.items():
            t = t.replace("{{" + k + "}}", str(v).replace("\\%", "%").replace("$\\pm$", "±").replace("--", "–"))
        left = re.findall(r"\{\{([A-Z0-9]+)\}\}", t)
        if left:
            raise SystemExit(f"unfilled {name} placeholders: {left}")
        out = {"response": "Response_to_Reviewers", "cover_letter": "Cover_Letter"}[name]
        open(os.path.join(LET, f"{out}.md"), "w").write(t)
        md_to_docx(os.path.join(LET, f"{out}.md"), os.path.join(LET, f"{out}.docx"))
        print(f"wrote {out}.md/.docx")
    json.dump(N, open(os.path.join(R4, "reported_numbers.json"), "w"), indent=2)


def main():
    N = compute_numbers()
    intro = open(os.path.join(PARTS, "intro_methods.tex")).read()
    body = open(os.path.join(PARTS, "body.tex")).read().replace("<<INTRO_METHODS>>", intro)
    for k, v in N.items():
        body = body.replace("\\PH{" + k + "}", v)
    left = re.findall(r"\\PH\{([A-Z0-9]+)\}", body)
    if left:
        raise SystemExit(f"unfilled manuscript placeholders: {sorted(set(left))}")
    pool = open(os.path.join(PARTS, "bib_pool.tex")).read()
    bib, nref = bibliography(body, pool)
    tex = PREAMBLE + body + "\n\\bibliographystyle{elsarticle-num}\n" + bib + "\n\\end{document}\n"
    open(os.path.join(SUB, "HiOmicsFormer_manuscript.tex"), "w").write(tex)
    json.dump(N, open(os.path.join(R4, "reported_numbers.json"), "w"), indent=2)
    print(f"manuscript written: {nref} references")


if __name__ == "__main__":
    import sys
    if "--letters" in sys.argv:
        build_letters(compute_numbers())
    else:
        main()
