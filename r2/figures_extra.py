"""
Regenerate the key data-dependent figures from the CORRECTED pipeline outputs, and
emit a figure freshness manifest so nothing stale is mistaken for current.

Regenerates: fig1 (dataset overview, 11 types/3044 pts), fig3 (t-SNE of frozen
embeddings), fig5 (biomarkers — now real gene symbols + method overlap).
Marks the remaining notebook figures (fig2/fig6/fig9) that require re-running notebook
plotting cells as STALE with the reason.
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from nbdefs import config
RES = config.results_path; FIG = config.figures_path; CACHE = os.path.join(HERE, "cache")


def fig1(meta):
    q = meta["quality_report"]; fd = meta["feature_dims"]
    # relabel THCA->COAD already applied in labels; use actual label counts
    cl = pd.Series(meta["cancer_labels"]).value_counts().sort_values(ascending=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.barh(cl.index, cl.values, color="#4575b4")
    for i, v in enumerate(cl.values): a.text(v + 2, i, str(v), va="center", fontsize=8)
    a.set_title(f"(a) {len(cl)} cancer types, {int(cl.sum())} patients (corrected)", fontweight="bold")
    a.set_xlabel("patients")
    mods = list(fd.keys()); vals = [fd[m] for m in mods]
    b.bar(mods, vals, color="#d73027")
    for i, v in enumerate(vals): b.text(i, v + 20, str(v), ha="center", fontsize=9)
    b.set_title("(b) selected features / modality (clean universe)", fontweight="bold")
    b.set_ylabel("features")
    fig.suptitle("Figure 1 (revised): Dataset overview after integrity corrections", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/fig1_dataset_overview.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig3(meta):
    from sklearn.manifold import TSNE
    emb = np.load(f"{RES}/best_fold_embeddings.npy")
    clus = np.load(f"{RES}/best_fold_clusters.npy")
    cl = np.asarray(meta["cancer_labels"])
    z2 = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(emb)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6))
    for t in sorted(set(cl)):
        m = cl == t; a.scatter(z2[m, 0], z2[m, 1], s=6, alpha=0.6, label=t)
    a.legend(fontsize=7, ncol=2, markerscale=2); a.set_title("(a) t-SNE by cancer type", fontweight="bold")
    for c in sorted(set(clus)):
        m = clus == c
        if m.sum(): b.scatter(z2[m, 0], z2[m, 1], s=6, alpha=0.6, label=f"C{c} (n={m.sum()})")
    b.legend(fontsize=7, markerscale=2)
    b.set_title(f"(b) t-SNE by DEC cluster ({len(set(clus[np.bincount(clus)[clus]>0]))} effective)",
                fontweight="bold")
    fig.suptitle("Figure 3 (revised): Frozen best-fold embeddings", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/fig3_training_embeddings.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig5():
    attn = json.load(open(f"{RES}/biomarkers_attention.json"))
    ov = pd.read_csv(f"{RES}/biomarker_method_overlap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    rows = []
    for mod in config.modalities:
        for g in attn.get(mod, [])[:6]:
            rows.append((f"{mod}:{g}", mod))
    y = np.arange(len(rows))
    colors = {m: c for m, c in zip(config.modalities, ["#4575b4", "#91bfdb", "#fc8d59", "#d73027"])}
    ax.barh(y, [1] * len(rows), color=[colors[m] for _, m in rows])
    ax.set_yticks(y); ax.set_yticklabels([r for r, _ in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_title("(a) Top attention biomarkers (real gene symbols — no ZZZ padding)",
                                    fontweight="bold", fontsize=10)
    ax2 = axes[1]
    x = np.arange(len(ov));
    ax2.bar(x, ov["overlap"], color="#2c7fb8", label="consensus (attn∩recon)")
    ax2.bar(x, ov["attn_n"] - ov["overlap"], bottom=ov["overlap"], color="#a6bddb", label="attn only")
    ax2.set_xticks(x); ax2.set_xticklabels(ov["modality"]); ax2.legend(fontsize=8)
    ax2.set_title("(b) Method agreement (attn vs recon, /50)", fontweight="bold", fontsize=10)
    fig.suptitle("Figure 5 (revised): Biomarkers are real symbols; low cross-method overlap quantified",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/fig5_biomarkers.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def manifest(run_start):
    rows = []
    for f in sorted(os.listdir(FIG)):
        if not f.endswith(".png"):
            continue
        mt = os.path.getmtime(os.path.join(FIG, f))
        rows.append({"figure": f, "regenerated_this_run": bool(mt >= run_start),
                     "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(mt))})
    json.dump(rows, open(f"{RES}/figure_manifest.json", "w"), indent=2)
    stale = [r["figure"] for r in rows if not r["regenerated_this_run"] and "CONTAMINATED" not in r["figure"]]
    return rows, stale


def main():
    run_start = time.time() - 5
    meta = pickle.load(open(f"{CACHE}/meta.pkl", "rb"))
    fig1(meta); print("fig1 ok", flush=True)
    fig3(meta); print("fig3 ok", flush=True)
    fig5(); print("fig5 ok", flush=True)
    rows, stale = manifest(run_start)
    print("STALE (need notebook plotting cells with corrected objects):", stale)


if __name__ == "__main__":
    main()
