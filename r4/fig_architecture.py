"""Architecture schematic of HiOmicsFormer AS IMPLEMENTED (r2/nbdefs.py). Vector PDF."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "HiOmicsFormer_Elsevier_Submission")

C = {"in": "#dbe9f6", "enc": "#fde5cc", "cl": "#e5f2e0", "att": "#f6dede", "fus": "#e8e1f3",
     "dec": "#fff3c4", "rec": "#eeeeee", "edge": "#444444"}


def box(ax, x, y, w, h, text, fc, fs=7.2, bold=False, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=C["edge"], lw=0.8))
    tx = x + w / 2 if ha == "center" else x + 0.015
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                                 lw=0.8, color=C["edge"], linestyle=ls))


def main():
    fig = plt.figure(figsize=(7.2, 7.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    mods = [("mRNA", "2000"), ("miRNA", "1434"), ("Methylation", "2000"), ("CNV", "2000")]
    xs = [0.03, 0.18, 0.33, 0.48]; w = 0.135
    for (m, p), x in zip(mods, xs):
        box(ax, x, 0.905, w, 0.06, f"{m}\n$x^{{(m)}}\\in\\mathbb{{R}}^{{{p}}}$", C["in"], bold=False)
        box(ax, x, 0.745, w, 0.12, "Multi-scale\nencoder\n(own weights)\n$e^{(m)}\\in\\mathbb{R}^{256}$", C["enc"])
        arrow(ax, x + w / 2, 0.905, x + w / 2, 0.865)
    # encoder detail panel
    box(ax, 0.655, 0.700, 0.325, 0.265,
        "Multi-scale encoder (per modality)\n"
        "1. Linear $p_m\\!\\rightarrow\\!256$ + LayerNorm + GELU\n"
        "2. Split the 256-d vector into chunks of\n    16, 32, 64 units (16, 8, 4 tokens)\n"
        "3. Per scale: project tokens to 256-d,\n    prepend [CLS], positional encoding,\n"
        "    4 pre-LN Transformer blocks (8 heads)\n"
        "4. Concatenate the 3 [CLS] outputs\n    $\\rightarrow$ MLP $\\rightarrow e^{(m)}$",
        "#fff8ef", fs=6.6, ha="left")
    arrow(ax, 0.615, 0.805, 0.655, 0.805, ls="--")
    # contrastive
    box(ax, 0.03, 0.615, 0.585, 0.085,
        "Cross-modal contrastive loss $\\mathcal{L}_{CL}$ (self-supervised): InfoNCE over the 6 modality pairs;\n"
        "positive = the same patient's other modality; per-modality projector $256\\!\\rightarrow\\!128$; $\\tau = 0.07$",
        C["cl"], fs=6.8)
    for x in xs:
        arrow(ax, x + w / 2, 0.745, x + w / 2, 0.700)
    # gated cross-modal attention
    box(ax, 0.03, 0.505, 0.585, 0.08,
        "Gated cross-modal attention ($\\times 2$ layers): each modality attends to all four\n"
        "embeddings; output scaled by a sigmoid gate, then residual + LayerNorm + feed-forward",
        C["att"], fs=6.8)
    arrow(ax, 0.3225, 0.615, 0.3225, 0.585)
    # fusion + latent
    box(ax, 0.03, 0.385, 0.585, 0.09,
        "Fusion: concatenate the 4 embeddings ($4\\times256$) $\\rightarrow$ Linear + LayerNorm + GELU $\\rightarrow$ 256\n"
        "Variational latent: $(\\mu, \\log\\sigma^2)\\in\\mathbb{R}^{128}$;  $z=\\mu+\\sigma\\odot\\epsilon$ in training, $z=\\mu$ at inference",
        C["fus"], fs=6.8)
    arrow(ax, 0.3225, 0.505, 0.3225, 0.475)
    # heads
    box(ax, 0.03, 0.235, 0.28, 0.115,
        "DEC head on $z$\nStudent-$t$ kernel ($\\alpha=1$) to $K_c$ centroids\n(K-means initialised after pre-training)\n"
        "$\\mathcal{L}_{DEC}=\\mathrm{KL}(P\\,\\|\\,Q)$ $\\rightarrow$ cluster = argmax $q$",
        C["dec"], fs=6.6)
    box(ax, 0.335, 0.235, 0.28, 0.115,
        "Decoders (one per modality)\n$z\\rightarrow256\\rightarrow p_m$\n$\\mathcal{L}_{recon}$ = mean squared error\n"
        "$\\mathcal{L}_{KL}$ = KL$(\\,q(z\\,|\\,x)\\,\\|\\,\\mathcal{N}(0,I))$",
        C["rec"], fs=6.6)
    arrow(ax, 0.17, 0.385, 0.17, 0.350); arrow(ax, 0.475, 0.385, 0.475, 0.350)
    # objective + protocol
    box(ax, 0.03, 0.045, 0.95, 0.15,
        "Objective: $\\mathcal{L}=1.0\\,\\mathcal{L}_{recon}+0.1\\,\\mathcal{L}_{CL}+0.5\\,w(t)\\,\\mathcal{L}_{DEC}+0.01\\,\\mathcal{L}_{KL}$, "
        "with $w(t)=\\min(1, t/10)$.\n"
        "Training: (1) pre-training without $\\mathcal{L}_{DEC}$ (up to 30 epochs); (2) K-means initialisation of the centroids;\n"
        "(3) joint training (up to 200 epochs). Both stages early-stop on the validation silhouette of $z$. AdamW, lr $10^{-4}$.\n"
        "Cancer-type labels are not used by any loss; they are used only to stratify cross-validation folds and to evaluate.",
        "#f7f7f7", fs=6.8)
    box(ax, 0.655, 0.235, 0.325, 0.44,
        "Implementation sizes\n\n"
        "hidden width $d$ = 256\nlatent dimension = 128\nTransformer blocks: 4 per scale\n    (3 scales per modality)\n"
        "attention heads = 8\ncross-modal layers = 2\ncontrastive projection = 128\n"
        "dropout = 0.1\nbatch size = 256\nweight decay = $10^{-5}$\n$K_c$ = number of cancer types\n    (11 full cohort; 8 single archive)",
        "#ffffff", fs=6.6, ha="left")
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "fig_architecture.pdf"), bbox_inches="tight")
    print("wrote", os.path.join(OUTDIR, "fig_architecture.pdf"))


if __name__ == "__main__":
    main()
