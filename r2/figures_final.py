"""
Phase B — regenerate all DATA figures for the manuscript, at 600 dpi, WITHOUT any
burned-in figure title (captions carry the number), as vector PDF written directly into
HiOmicsFormer_Elsevier_Submission/. Panels trimmed to those that carry an argument.

Figure map (LaTeX auto-numbers by \begin{figure} order; filenames need not match):
  fig1_dataset_overview  fig2_preprocessing  fig3_training_embeddings
  fig4_baseline_comparison  fig5_biomarkers  fig7b_stratified_survival(copied)
  fig8_imputation_sensitivity  fig11_cluster_composition(copied)
Enrichment (fig6) and multi-seed subanalysis (fig9) are NOT regenerated (see RESULTS_SUMMARY);
the old pan-cancer survival (fig7) is superseded by fig7b.
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600, "font.size": 10,
                     "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, HERE)
RES = "results_revision"; SUB = "HiOmicsFormer_Elsevier_Submission"; CACHE = "r2/cache"
meta = pickle.load(open(f"{CACHE}/meta.pkl", "rb"))
kn = json.load(open(f"{RES}/r2_key_numbers.json"))
PAL = plt.cm.tab20.colors


def save(fig, name):
    fig.savefig(f"{SUB}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{SUB}/{name}.pdf", flush=True)


def f_dataset():
    cl = pd.Series(meta["cancer_labels"]).value_counts().sort_values()
    perm = kn["imputation_permod"]
    mods = ["mRNA", "miRNA", "Methy", "CNV"]
    raw = [perm[m]["raw_missing"] * 100 for m in mods]; eff = [perm[m]["eff_missing"] * 100 for m in mods]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].barh(cl.index, cl.values, color="#4575b4")
    for i, v in enumerate(cl.values): ax[0].text(v + 2, i, str(v), va="center", fontsize=8)
    ax[0].set_xlabel("patients"); ax[0].set_title(f"(a) {len(cl)} cancer types, N={int(cl.sum())}")
    x = np.arange(len(mods)); w = 0.38
    ax[1].bar(x - w/2, raw, w, label="raw", color="#d73027")
    ax[1].bar(x + w/2, eff, w, label="effective (post-filter)", color="#fc8d59")
    ax[1].set_xticks(x); ax[1].set_xticklabels(mods); ax[1].set_ylabel("% missing")
    ax[1].set_title("(b) missingness (raw vs effective)"); ax[1].legend(fontsize=8)
    fd = kn["feature_dims_selected"]
    ax[2].bar(mods, [fd[m] for m in mods], color="#1a9850")
    for i, m in enumerate(mods): ax[2].text(i, fd[m] + 20, str(fd[m]), ha="center", fontsize=8)
    ax[2].set_ylabel("selected features"); ax[2].set_title("(c) features / modality (clean universe)")
    fig.tight_layout(); save(fig, "fig1_dataset_overview")


def f_preprocessing():
    # raw union vs selected, per modality (from cleaning: raw pools known)
    raw = {"mRNA": 21886, "miRNA": 1434, "Methy": 19087, "CNV": 19565}  # clean-universe pools (phase0 log)
    fd = kn["feature_dims_selected"]; mods = list(fd.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(mods)); w = 0.38
    ax[0].bar(x - w/2, [raw[m] for m in mods], w, label="clean universe", color="#91bfdb")
    ax[0].bar(x + w/2, [fd[m] for m in mods], w, label="MAD-selected", color="#4575b4")
    ax[0].set_xticks(x); ax[0].set_xticklabels(mods); ax[0].set_ylabel("features"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) feature selection (post name-harmonization)")
    ax[1].axis("off")
    tab = [[m, f"{raw[m]:,}", f"{fd[m]:,}"] for m in mods] + [["total", "", f"{sum(fd.values()):,}"]]
    t = ax[1].table(cellText=tab, colLabels=["modality", "clean universe", "selected"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
    ax[1].set_title("(b) processed dimensions")
    fig.tight_layout(); save(fig, "fig2_preprocessing")


def f_embeddings():
    from sklearn.manifold import TSNE
    emb = np.load(f"{RES}/best_fold_embeddings.npy"); clus = np.load(f"{RES}/best_fold_clusters.npy")
    cl = np.asarray(meta["cancer_labels"])
    z2 = TSNE(2, perplexity=30, random_state=42, init="pca").fit_transform(emb)
    from sklearn.decomposition import PCA
    pcs = PCA(10).fit(emb); evr = pcs.explained_variance_ratio_
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for t in sorted(set(cl)):
        m = cl == t; ax[0].scatter(z2[m, 0], z2[m, 1], s=6, alpha=0.6, label=t)
    ax[0].legend(fontsize=6, ncol=2, markerscale=2); ax[0].set_title("(a) t-SNE by cancer type")
    ax[0].set_xlabel("t-SNE 1"); ax[0].set_ylabel("t-SNE 2")
    for c in sorted(set(clus)):
        m = clus == c
        if m.sum(): ax[1].scatter(z2[m, 0], z2[m, 1], s=6, alpha=0.6, label=f"C{c} (n={m.sum()})")
    ax[1].legend(fontsize=7, markerscale=2); ax[1].set_title("(b) t-SNE by DEC cluster (2 effective)")
    ax[1].set_xlabel("t-SNE 1"); ax[1].set_ylabel("t-SNE 2")
    ax[2].bar(range(1, 11), evr * 100, color="#d73027")
    ax[2].set_xlabel("principal component"); ax[2].set_ylabel("% variance")
    ax[2].set_title(f"(c) embedding PCA (PC1={evr[0]*100:.1f}%)")
    fig.tight_layout(); save(fig, "fig3_training_embeddings")


def f_baseline():
    df = pd.read_csv(f"{RES}/baseline_comparison.csv").sort_values("silhouette_mean")
    y = np.arange(len(df))
    fig, ax = plt.subplots(1, 3, figsize=(15, 0.42 * len(df) + 1.5), sharey=True)
    for a, met, ti in zip(ax, ["silhouette", "nmi", "ari"], ["Silhouette", "NMI", "ARI"]):
        col = ["#d73027" if m == "HiOmicsFormer" else ("#4575b4" if m == "SimpleMultiOmicsTransformer" else "#999999") for m in df["method"]]
        a.barh(y, df[f"{met}_mean"], xerr=df[f"{met}_std"], color=col, capsize=2)
        a.set_yticks(y); a.set_yticklabels(df["method"], fontsize=8); a.set_xlabel(f"{ti} (mean±SD)")
        a.set_title(f"({'abc'[['silhouette','nmi','ari'].index(met)]}) {ti}")
    fig.tight_layout(); save(fig, "fig4_baseline_comparison")


def f_biomarkers():
    attn = json.load(open(f"{RES}/biomarkers_attention.json")); ov = pd.read_csv(f"{RES}/biomarker_method_overlap.csv")
    mods = ["mRNA", "miRNA", "Methy", "CNV"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = [(f"{m}:{g}", m) for m in mods for g in attn.get(m, [])[:6]]
    y = np.arange(len(rows)); cmap = dict(zip(mods, ["#4575b4", "#91bfdb", "#fc8d59", "#d73027"]))
    ax[0].barh(y, [1]*len(rows), color=[cmap[m] for _, m in rows])
    ax[0].set_yticks(y); ax[0].set_yticklabels([r for r, _ in rows], fontsize=7); ax[0].invert_yaxis()
    ax[0].set_xticks([]); ax[0].set_title("(a) top attention-gradient features (real identifiers)")
    x = np.arange(len(ov))
    ax[1].bar(x, ov["overlap"], color="#2c7fb8", label="consensus (attn∩recon)")
    ax[1].bar(x, ov["attn_n"] - ov["overlap"], bottom=ov["overlap"], color="#a6bddb", label="attention only")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ov["modality"]); ax[1].legend(fontsize=8)
    ax[1].set_ylabel("features (/50)"); ax[1].set_title("(b) attention vs reconstruction overlap")
    fig.tight_layout(); save(fig, "fig5_biomarkers")


def f_imputation():
    df = pd.read_csv(f"{RES}/imputation_sensitivity_fixed.csv")
    piv = df.pivot_table(index="modality", columns="pair", values="jaccard", aggfunc="mean")
    piv = piv.reindex(["mRNA", "miRNA", "Methy", "CNV"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax[0].imshow(piv.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax[0].set_xticks(range(piv.shape[1])); ax[0].set_xticklabels(piv.columns, rotation=25, ha="right", fontsize=8)
    ax[0].set_yticks(range(piv.shape[0])); ax[0].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v): ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
    ax[0].set_title("(a) Jaccard(top-50) between imputers — MAD on unstandardized matrix")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="Jaccard")
    rig = kn["imputation_rigor"]; mods = list(piv.index)
    miss = [kn["imputation_permod"][m]["raw_missing"] * 100 for m in mods]
    top50 = [rig[m]["top50_missingness_max"] * 100 for m in mods]
    y = np.arange(len(mods))
    ax[1].barh(y - 0.2, miss, 0.4, label="modality raw % missing", color="#bdbdbd")
    ax[1].barh(y + 0.2, top50, 0.4, label="top-50 features % missing", color="#2c7fb8")
    ax[1].set_yticks(y); ax[1].set_yticklabels(mods); ax[1].set_xlabel("% missing"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) selected features are ~0% missing\n(chance Jaccard ≈ 0.013)")
    fig.tight_layout(); save(fig, "fig8_imputation_sensitivity")


def f_cluster_composition():
    comp = pd.read_csv(f"{RES}/cluster_validity_composition.csv", index_col=0)
    rn = comp.div(comp.sum(1).replace(0, np.nan), axis=0).fillna(0)
    cv = kn["cluster_validity"]
    fig, ax = plt.subplots(figsize=(1.0 * rn.shape[1] + 2, 0.5 * rn.shape[0] + 2))
    im = ax.imshow(rn.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rn.shape[1])); ax.set_xticklabels(rn.columns, rotation=45, ha="right")
    ax.set_yticks(range(rn.shape[0])); ax.set_yticklabels([f"C{c}" for c in rn.index])
    for i in range(rn.shape[0]):
        for j in range(rn.shape[1]):
            v = rn.values[i, j]
            if v >= 0.10: ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    ax.set_xlabel("cancer type"); ax.set_ylabel("DEC cluster")
    ax.set_title(f"purity={cv['overall_purity']:.2f}  NMI={cv['nmi']:.2f}  ARI={cv['ari']:.2f}  "
                 f"Cramér's V(type)={cv['cramers_v']:.2f} < V(source)={cv['technical_source']['cramers_v']:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of cluster")
    fig.tight_layout(); save(fig, "fig11_cluster_composition")


def f_stratified_survival():
    # rebuild titleless KM panels from stratified_survival.csv + embeddings
    from sklearn.cluster import KMeans
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    emb = np.load(f"{RES}/best_fold_embeddings.npy"); cl = np.asarray(meta["cancer_labels"])
    sd = meta["survival_data"].copy()
    sd = sd[~sd.index.duplicated(keep="first")]
    ids = [s.split("_", 1)[1] for s in meta["sample_ids"]]
    res = pd.read_csv(f"{RES}/stratified_survival.csv")
    ok = res[res["note"] == "ok"].sort_values("logrank_p")["cancer_type"].tolist()[:6]
    n = len(ok); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.3*nrow), squeeze=False)
    for ax, t in zip(axes.ravel(), ok):
        idx = np.where(cl == t)[0]
        have = [(i, ids[i]) for i in idx if ids[i] in sd.index]
        rid = [i for i, _ in have]; sids = [s for _, s in have]
        km = KMeans(2, random_state=42, n_init=10).fit_predict(emb[rid])
        surv = sd.loc[sids, ["survival_times", "event_observed"]].apply(pd.to_numeric, errors="coerce")
        surv["s"] = km; surv = surv.dropna()
        kmf = KaplanMeierFitter()
        for s in [0, 1]:
            m = surv.s == s
            if m.sum() >= 10:
                kmf.fit(surv.loc[m, "survival_times"], surv.loc[m, "event_observed"], label=f"subtype {s} (n={m.sum()})")
                kmf.plot_survival_function(ax=ax, ci_show=False)
        p = float(res[res.cancer_type == t]["logrank_p"].iloc[0])
        ax.set_title(f"{t} (log-rank p={p:.3f})", fontsize=9); ax.set_xlabel("time"); ax.set_ylabel("survival"); ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]: ax.axis("off")
    fig.tight_layout(); save(fig, "fig7b_stratified_survival")


for f in [f_dataset, f_preprocessing, f_embeddings, f_baseline, f_biomarkers,
          f_imputation, f_cluster_composition, f_stratified_survival]:
    try:
        f()
    except Exception as e:
        print(f"  !! {f.__name__} failed: {e}", flush=True)
print("done")
