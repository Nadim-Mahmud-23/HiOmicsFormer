"""Main and supplementary figures for the revised manuscript, from results_revision/ artifacts (vector PDF)."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, OUT, config, load_cache, archive_of

SUB = os.path.join(ROOT, "HiOmicsFormer_Elsevier_Submission")
os.makedirs(SUB, exist_ok=True)
RES = os.path.join(ROOT, "results_revision")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False})
ARCH_COL = {0: "#4575b4", 1: "#d73027"}
ARCH_NAME = {0: "clustering archive", 1: "classification archive"}


def J(name):
    return json.load(open(os.path.join(OUT, name)))


def save(fig, name):
    fig.savefig(os.path.join(SUB, f"{name}.pdf"), bbox_inches="tight"); plt.close(fig); print("wrote", name)


def panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def fig_cohort():
    _, meta, _ = load_cache("all")
    lab = pd.Series([str(x) for x in meta["cancer_labels"]])
    cnt = lab.value_counts().sort_values()
    arch = archive_of(cnt.index)
    miss = J("missingness.json")["all"]["selected_panel"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax[0].barh(cnt.index, cnt.values, color=[ARCH_COL[a] for a in arch])
    for i, v in enumerate(cnt.values):
        ax[0].text(v + 5, i, str(v), va="center", fontsize=6.5)
    ax[0].set_xlabel("patients"); ax[0].set_title(f"Cohort: {int(cnt.sum()):,} patients, {len(cnt)} types")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ARCH_COL[a]) for a in (0, 1)],
                 labels=["clustering archive (survival data)", "classification archive"], loc="upper center",
                 bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False)
    panel(ax[0], "a")
    mods = config.modalities; x = np.arange(len(mods)); w = 0.38
    c0 = [miss[m]["missing_fraction_clustering_archive"] * 100 for m in mods]
    c1 = [miss[m]["missing_fraction_classification_archive"] * 100 for m in mods]
    ax[1].bar(x - w / 2, c0, w, color=ARCH_COL[0], label=ARCH_NAME[0])
    ax[1].bar(x + w / 2, c1, w, color=ARCH_COL[1], label=ARCH_NAME[1])
    for xi, v in zip(np.r_[x - w / 2, x + w / 2], c0 + c1):
        ax[1].text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["mRNA", "miRNA", "Methylation", "CNV"])
    ax[1].set_ylabel("% of input values imputed")
    ax[1].set_title("Imputed values, by archive"); ax[1].legend(frameon=False)
    panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig1_cohort")


def fig_full_cohort():
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    _, meta, _ = load_cache("all")
    lab = np.asarray([str(x) for x in meta["cancer_labels"]])
    emb = np.load(os.path.join(RES, "all_uncorrected", "embeddings.npy"))
    clu = np.load(os.path.join(RES, "all_uncorrected", "clusters.npy"))
    z2 = TSNE(2, perplexity=30, random_state=42, init="pca").fit_transform(emb)
    evr = PCA(10).fit(emb).explained_variance_ratio_ * 100
    fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    cmap = plt.cm.tab20.colors
    for i, t in enumerate(sorted(set(lab))):
        m = lab == t; a0.scatter(z2[m, 0], z2[m, 1], s=2, color=cmap[i * 2 % 20], label=t, rasterized=True)
    a0.legend(fontsize=5.3, ncol=4, markerscale=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
               handletextpad=0.1, columnspacing=0.6); a0.set_title("t-SNE, by cancer type")
    sizes = pd.Series(clu).value_counts()
    for i, c in enumerate(sizes.index):
        m = clu == c; a1.scatter(z2[m, 0], z2[m, 1], s=2, color=plt.cm.Set1.colors[i % 9], label=f"C{c} (n={m.sum()})", rasterized=True)
    a1.legend(fontsize=6, markerscale=3, frameon=False); a1.set_title("t-SNE, by DEC cluster")
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("t-SNE 1"); a.set_ylabel("t-SNE 2")
    a2.bar(range(1, 11), evr, color="#636363"); a2.set_xlabel("principal component"); a2.set_ylabel("% variance")
    a2.set_title(f"PCA of embedding\n(PC1 = {evr[0]:.1f}%)"); a2.set_xticks(range(1, 11))
    panel(a0, "a"); panel(a1, "b"); panel(a2, "c")
    a3 = fig.add_subplot(gs[1, :])
    keep = sizes.index
    tab = pd.crosstab(pd.Series(clu, name="cluster"), pd.Series(lab, name="type")).loc[keep]
    rown = tab.div(tab.sum(1), axis=0)
    im = a3.imshow(rown.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    a3.set_xticks(range(tab.shape[1])); a3.set_xticklabels(tab.columns)
    a3.set_yticks(range(tab.shape[0])); a3.set_yticklabels([f"C{c} (n={tab.loc[c].sum()})" for c in tab.index])
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            if tab.values[i, j]:
                a3.text(j, i, str(tab.values[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if rown.values[i, j] > 0.5 else "black")
    for j, t in enumerate(tab.columns):
        if t in ("GBM", "LGG", "OV"):
            a3.get_xticklabels()[j].set_color(ARCH_COL[1])
    a3.set_title("Cluster composition (patients; shading = fraction of the cluster; red = classification-archive types)")
    fig.colorbar(im, ax=a3, fraction=0.02, pad=0.01, label="row fraction")
    panel(a3, "d")
    fig.tight_layout(w_pad=2.5); save(fig, "fig3_full_cohort")


def fig_archive_axis():
    from sklearn.decomposition import PCA
    _, meta, _ = load_cache("all")
    lab = np.asarray([str(x) for x in meta["cancer_labels"]]); arch = archive_of(lab)
    emb = np.load(os.path.join(RES, "all_uncorrected", "embeddings.npy"))
    pc1 = PCA(2).fit_transform(emb)[:, 0]
    if pc1[arch == 1].mean() < pc1[arch == 0].mean():
        pc1 = -pc1
    dp = J("duplicate_patient_test.json"); df = pd.read_csv(os.path.join(OUT, "duplicate_patient_pc1.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1, 1.0]})
    bins = np.linspace(pc1.min(), pc1.max(), 50)
    for a in (0, 1):
        ax[0].hist(pc1[arch == a], bins=bins, color=ARCH_COL[a], alpha=0.8, label=ARCH_NAME[a])
    ax[0].set_xlabel("embedding PC1"); ax[0].set_ylabel("patients"); ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title("Full cohort"); panel(ax[0], "a")
    p1 = df[df["copy"] == "clustering_copy"].set_index("barcode")["pc1"]
    p2 = df[df["copy"] == "classification_copy"].set_index("barcode")["pc1"].reindex(p1.index)
    for a, b in zip(p1.values, p2.values):
        ax[1].plot([0, 1], [a, b], color="#bbbbbb", lw=0.4, zorder=1)
    ax[1].scatter(np.zeros(len(p1)), p1.values, s=5, color=ARCH_COL[0], zorder=2)
    ax[1].scatter(np.ones(len(p2)), p2.values, s=5, color=ARCH_COL[1], zorder=2)
    fm = dp["frozen_model"]
    for k, a in (("cohort_pc1_mean_clustering_archive", 0), ("cohort_pc1_mean_classification_archive", 1)):
        ax[1].axhline(fm[k], color=ARCH_COL[a], ls="--", lw=0.8)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["clustering-\narchive copy", "classification-\narchive copy"])
    ax[1].set_xlim(-0.4, 1.4); ax[1].set_ylabel("embedding PC1")
    ax[1].text(1.38, fm["cohort_pc1_mean_classification_archive"], "classification-\narchive mean", fontsize=5.5, va="center", ha="right", color=ARCH_COL[1])
    ax[1].text(1.38, fm["cohort_pc1_mean_clustering_archive"] + 0.9, "clustering-archive mean", fontsize=5.5, va="center", ha="right", color=ARCH_COL[0])
    ax[1].set_title("Same patients, two archives"); panel(ax[1], "b")
    mf = dp["model_free"]; mods = config.modalities; x = np.arange(len(mods))
    aucs = [mf[m]["archive_auc_patient_grouped_cv"] for m in mods]
    ax[2].bar(x, aucs, 0.6, color="#756bb1")
    ax[2].axhline(0.5, color="black", ls="--", lw=0.7); ax[2].text(3.45, 0.505, "chance", fontsize=6, ha="right", va="bottom")
    for i, m in enumerate(mods):
        ax[2].text(i, aucs[i] + 0.01, f"{aucs[i]:.2f}", ha="center", fontsize=6.5)
        ax[2].text(i, 0.08, f"r={mf[m]['median_corr_same_patient_across_archives']:.3f}", ha="center", fontsize=5.5, color="white", rotation=90)
    ax[2].set_xticks(x); ax[2].set_xticklabels(["mRNA", "miRNA", "Methyl.", "CNV"]); ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("copy-classifier AUC")
    ax[2].set_title("Input data (no model)"); panel(ax[2], "c")
    fig.tight_layout(w_pad=2.0); save(fig, "fig4_archive_axis")


def fig_clean_cohort():
    bs = pd.read_csv(os.path.join(OUT, "baselines_single_archive.csv"))
    ba = pd.read_csv(os.path.join(OUT, "baselines_all.csv"))
    sw = pd.read_csv(os.path.join(RES, "dec_collapse_sweep.csv"))
    ks = pd.read_csv(os.path.join(OUT, "known_subtypes.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.45, 1, 1.1]})
    order = bs.sort_values("nmi_mean")["method"].tolist()
    y = np.arange(len(order)); b = bs.set_index("method").loc[order]; a = ba.set_index("method").reindex(order)
    ax[0].barh(y + 0.2, a["nmi_mean"], 0.38, color="#bdbdbd", label="full cohort (11 types)")
    ax[0].barh(y - 0.2, b["nmi_mean"], 0.38, xerr=b["nmi_sd"], color=["#d73027" if m == "HiOmicsFormer" else "#4575b4" for m in order],
               label="single-archive cohort (8 types)", error_kw={"lw": 0.6})
    ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=6.3); ax[0].set_xlabel("NMI vs cancer type (test folds)")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#bdbdbd"), plt.Rectangle((0, 0), 1, 1, color="#4575b4"),
                          plt.Rectangle((0, 0), 1, 1, color="#d73027")],
                 labels=["full cohort (11 types)", "single-archive cohort (8 types)", "HiOmicsFormer, single archive"],
                 frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.3, -0.16), ncol=1)
    ax[0].set_title("Five-fold CV"); panel(ax[0], "a")
    s = sw[sw["note"].isna()].sort_values("Kc")
    ax[1].plot(s["Kc"], s["test_silhouette"], "o-", color="#636363", ms=3, label="silhouette (latent)")
    ax[1].plot(s["Kc"], s["test_nmi"], "s-", color="#d73027", ms=3, label="NMI vs cancer type")
    ax[1].set_xlabel("number of DEC centroids $K_c$"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, fontsize=6)
    ax[1].set_title("Cluster-count sweep"); panel(ax[1], "b")
    k = ks[ks["reference"] == "PanCanAtlas molecular subtype"].copy()
    k = k[((k["model"] == "single-archive model")) | ((k["model"] == "full-cohort model") & k["cancer_type"].isin(["GBM", "LGG", "OV"]))]
    k = k.sort_values("ARI_input_PCA")
    yy = np.arange(len(k))
    ax[2].scatter(k["ARI_input_PCA"], yy, color="#4575b4", s=14, label="input PCA + K-means")
    ax[2].scatter(k["ARI_model"], yy, color="#d73027", s=14, marker="D", label="HiOmicsFormer embedding")
    for i, (u, v) in enumerate(zip(k["ARI_input_PCA"], k["ARI_model"])):
        ax[2].plot([u, v], [i, i], color="#cccccc", lw=0.8, zorder=0)
    ax[2].axvline(0, color="black", lw=0.5)
    ax[2].set_yticks(yy); ax[2].set_yticklabels(k["cancer_type"]); ax[2].set_xlabel("ARI vs PanCanAtlas subtype")
    ax[2].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=1)
    ax[2].set_xlabel("ARI vs PanCanAtlas subtypes"); ax[2].set_title("Known subtypes"); panel(ax[2], "c")
    fig.tight_layout(); save(fig, "fig5_clean_cohort")


def figS_baselines_full():
    ba = pd.read_csv(os.path.join(OUT, "baselines_all.csv")).sort_values("silhouette_mean")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    y = np.arange(len(ba))
    for a, met, ti in zip(ax, ("silhouette", "nmi", "ari"), ("Silhouette (own space)", "NMI", "ARI")):
        a.barh(y, ba[f"{met}_mean"], xerr=ba[f"{met}_sd"], color=["#d73027" if m == "HiOmicsFormer" else "#999999" for m in ba["method"]],
               error_kw={"lw": 0.6})
        a.set_title(ti); a.axvline(0, color="black", lw=0.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(ba["method"], fontsize=6.3)
    fig.tight_layout(); save(fig, "figS1_baselines_full_cohort")


def figS_missing_by_type():
    t = pd.read_csv(os.path.join(OUT, "missingness_by_type.csv"))
    t = t[t["cohort"] == "all"].pivot(index="cancer_type", columns="modality", values="missing_fraction_selected")[config.modalities] * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(t.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["mRNA", "miRNA", "Methyl.", "CNV"]); ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if t.values[i, j] > 50 else "black")
    fig.colorbar(im, ax=ax, label="% imputed"); ax.set_title("Imputed share of the selected features")
    fig.tight_layout(); save(fig, "figS2_missing_by_type")


def figS_feature_scores():
    sc = pd.read_csv(os.path.join(OUT, "feature_scores_all.csv"))
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6))
    for a, m in zip(ax, config.modalities):
        d = sc[sc["modality"] == m].sort_values("gradient", ascending=False).head(10)[::-1]
        a.barh(d["feature"].astype(str), d["gradient"] / d["gradient"].max(), color="#756bb1")
        a.set_title(m); a.set_xlabel("relative gradient score"); a.tick_params(axis="y", labelsize=6)
    fig.tight_layout(); save(fig, "figS5_feature_scores")


def figS_imputation():
    df = pd.read_csv(os.path.join(RES, "imputation_sensitivity_fixed.csv"))
    piv = df.pivot_table(index="modality", columns="pair", values="jaccard").reindex(config.modalities)
    null = df.groupby("modality")["null_jaccard"].first().reindex(config.modalities)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    im = ax.imshow(piv.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{m} (chance {null[m]:.3f})" for m in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="Jaccard (top 50)"); ax.set_title("Top-50 features by MAD: overlap between imputation methods")
    fig.tight_layout(); save(fig, "figS3_imputation_stability")


def figS_km():
    from sklearn.cluster import KMeans
    from lifelines import KaplanMeierFitter
    from common import SEED
    _, meta, _ = load_cache("single_archive")
    lab = np.asarray([str(x) for x in meta["cancer_labels"]])
    emb = np.load(os.path.join(RES, "single_archive", "embeddings.npy"))
    sd = meta["survival_data"]; sd = sd[~sd.index.duplicated(keep="first")]
    ids = [x.split("_", 1)[1] for x in meta["sample_ids"]]
    res = pd.read_csv(os.path.join(RES, "single_archive", "stratified_survival.csv")).set_index("cancer_type")
    types = sorted(set(lab)); fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharey=True)
    for ax, t in zip(axes.flat, types):
        idx = [i for i in np.where(lab == t)[0] if ids[i] in sd.index]
        km2 = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(emb[idx])
        s = sd.loc[[ids[i] for i in idx], ["survival_times", "event_observed"]].apply(pd.to_numeric, errors="coerce")
        s["c"] = km2; s = s.dropna()
        for c, col in ((0, "#4575b4"), (1, "#d73027")):
            g = s[s.c == c]
            KaplanMeierFitter().fit(g.survival_times / 365.25, g.event_observed, label=f"cluster {c + 1} (n={len(g)})").plot_survival_function(
                ax=ax, ci_show=False, color=col, lw=1)
        p = res.loc[t, "logrank_p"]; q = res.loc[t, "logrank_fdr"]
        ax.set_title(f"{t}: p={p:.2f}, FDR={q:.2f}", fontsize=7.5); ax.set_xlabel("years", fontsize=7); ax.legend(fontsize=5.5, frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("overall survival", fontsize=7)
    fig.tight_layout(); save(fig, "figS4_survival_km")


if __name__ == "__main__":
    for f in (fig_cohort, fig_full_cohort, fig_archive_axis, fig_clean_cohort, figS_baselines_full,
              figS_missing_by_type, figS_feature_scores, figS_imputation, figS_km):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skip {f.__name__}: {e}")
