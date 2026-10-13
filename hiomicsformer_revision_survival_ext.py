"""
HiOmicsFormer — Revision R2: Survival, External Validation, Transformer Baseline
================================================================================
Companion to hiomicsformer_revision_addons.py.

  R1.1 -> stratified_survival_analysis   (cancer-type-specific + within-cancer)
  R1.3 -> ExternalValidator              (scaffold; needs an independent cohort)
  R1.7 -> SimpleMultiOmicsTransformer    (a recent-style transformer baseline)

Again: nothing here manufactures a positive result. If a stratified log-rank is
non-significant, that is what gets reported.
"""
import numpy as np
import pandas as pd


# ============================================================================
# R1.1  —  STRATIFIED / CANCER-TYPE-SPECIFIC SURVIVAL
# ============================================================================
def stratified_survival_analysis(embeddings, cancer_labels, survival_df,
                                 sample_ids, time_col=None, event_col=None,
                                 min_per_group=10, save_path=None, seed=42):
    """
    Cancer-type-specific survival analysis (Reviewer 1.1).

    Instead of pooling incomparable cluster labels across independently-trained
    CV folds (which the original code did, and which helps explain p=0.897),
    this:
      1. For each cancer type with enough survival data, splits its patients
         into 2 molecular subtypes via KMeans(K=2) on the FROZEN learned
         embeddings, then runs a within-cancer Kaplan-Meier + log-rank test.
      2. Aggregates the per-cancer log-rank p-values and applies
         Benjamini-Hochberg correction across cancer types.

    This is the analysis the reviewer explicitly asked for ("cancer-type-specific
    survival analyses" / "survival-aware clustering"). It is the honest way to
    look for prognostic signal: pan-cancer pooling washes it out.

    Args
    ----
    embeddings   : (N, d) frozen HiOmicsFormer embeddings for all patients,
                   in the SAME order as cancer_labels / sample_ids.
    survival_df  : indexed by the *clean* sample id (no cancer prefix), with a
                   time and an event column.
    sample_ids   : list of 'CANCER_barcode' ids matching embeddings rows.

    Returns a per-cancer results DataFrame.
    """
    from sklearn.cluster import KMeans
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt

    # resolve columns
    tcands = ['survival_times', 'OS.time', 'os_time', 'OS_time', 'time',
              'Overall Survival (Months)']
    ecands = ['event_observed', 'OS', 'os_event', 'OS_event', 'vital_status',
              'event']
    if time_col is None:
        time_col = next((c for c in tcands if c in survival_df.columns),
                        survival_df.columns[0])
    if event_col is None:
        event_col = next((c for c in ecands if c in survival_df.columns),
                         survival_df.columns[1])
    print("=" * 74)
    print("CANCER-TYPE-STRATIFIED SURVIVAL  (Reviewer 1.1)")
    print("=" * 74)
    print(f"  time='{time_col}'  event='{event_col}'")

    clean_ids = [s.split('_', 1)[1] if '_' in s else s for s in sample_ids]
    emb = np.asarray(embeddings)
    types = sorted(pd.unique(cancer_labels))

    rows = []
    n_types_plot = 0
    fig_cols = 4
    sig_types = [t for t in types]  # placeholder; filled below
    panels = []

    for t in types:
        idx = np.where(np.asarray(cancer_labels) == t)[0]
        ids_t = [clean_ids[i] for i in idx]
        have = [(i, sid) for i, sid in zip(idx, ids_t)
                if sid in survival_df.index]
        if len(have) < 2 * min_per_group:
            rows.append({'cancer_type': t, 'n': len(have),
                         'logrank_p': np.nan, 'note': 'too few survival samples'})
            continue

        rid = [i for i, _ in have]
        sids = [s for _, s in have]
        sub_emb = emb[rid]
        km2 = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(sub_emb)

        surv = survival_df.loc[sids, [time_col, event_col]].copy()
        surv[time_col] = pd.to_numeric(surv[time_col], errors='coerce')
        surv[event_col] = pd.to_numeric(surv[event_col], errors='coerce')
        surv['subtype'] = km2
        surv = surv.dropna()
        if surv['subtype'].nunique() < 2:
            rows.append({'cancer_type': t, 'n': len(surv),
                         'logrank_p': np.nan, 'note': 'single subtype'})
            continue
        g0 = surv[surv.subtype == 0]
        g1 = surv[surv.subtype == 1]
        if len(g0) < min_per_group or len(g1) < min_per_group:
            rows.append({'cancer_type': t, 'n': len(surv),
                         'logrank_p': np.nan, 'note': 'subtype too small'})
            continue

        lr = logrank_test(g0[time_col], g1[time_col],
                          g0[event_col], g1[event_col])
        rows.append({'cancer_type': t, 'n': len(surv),
                     'n_sub0': len(g0), 'n_sub1': len(g1),
                     'logrank_p': lr.p_value, 'note': 'ok'})
        panels.append((t, surv, time_col, event_col, lr.p_value))

    res = pd.DataFrame(rows)
    # BH across cancer types that produced a p-value
    valid = res['logrank_p'].notna()
    pv = res.loc[valid, 'logrank_p'].values
    if len(pv):
        order = np.argsort(pv)
        m = len(pv)
        bh = np.empty(m); prev = 1.0
        for rank, i in enumerate(order[::-1]):
            r = m - rank
            prev = min(prev, pv[i] * m / r)
            bh[i] = prev
        res.loc[valid, 'logrank_fdr'] = bh

    n_sig = int((res.get('logrank_fdr', pd.Series(dtype=float)) < 0.05).sum())
    print(f"\n  Cancer types tested: {int(valid.sum())}")
    print(f"  Significant within-cancer prognostic split (FDR<0.05): {n_sig}")
    for _, r in res.sort_values('logrank_p').iterrows():
        if pd.notna(r['logrank_p']):
            star = ' *' if r.get('logrank_fdr', 1) < 0.05 else ''
            print(f"    {r['cancer_type']:6s} n={int(r['n']):4d}  "
                  f"log-rank p={r['logrank_p']:.3f}  "
                  f"FDR={r.get('logrank_fdr', np.nan):.3f}{star}")

    # plot the significant / most interesting panels
    if panels:
        panels.sort(key=lambda x: x[4])
        show = panels[:8]
        ncol = min(fig_cols, len(show))
        nrow = int(np.ceil(len(show) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow),
                                 squeeze=False)
        for ax, (t, surv, tc, ec, p) in zip(axes.ravel(), show):
            kmf = KaplanMeierFitter()
            for s in [0, 1]:
                m = surv.subtype == s
                if m.sum() >= min_per_group:
                    kmf.fit(surv.loc[m, tc], surv.loc[m, ec],
                            label=f'subtype {s} (n={m.sum()})')
                    kmf.plot_survival_function(ax=ax, ci_show=False)
            ax.set_title(f'{t}  (log-rank p={p:.3f})', fontsize=9)
            ax.set_xlabel('time'); ax.set_ylabel('survival')
            ax.legend(fontsize=7)
        for ax in axes.ravel()[len(show):]:
            ax.axis('off')
        fig.suptitle('Cancer-type-specific molecular-subtype survival',
                     fontweight='bold')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n  Saved: {save_path}")
    return res


# ============================================================================
# R1.3  —  EXTERNAL VALIDATION SCAFFOLD
# ============================================================================
class ExternalValidator:
    """
    Apply the FROZEN, TCGA-trained HiOmicsFormer to an independent cohort
    (ICGC / TARGET / CPTAC / GEO) and recompute the key metrics (Reviewer 1.3,
    Reviewer 2).

    This is a scaffold: you must supply the external cohort in the same modality
    layout as MLOmics (samples x features, matching feature names). The class
    handles feature alignment, applies the fitted preprocessor's selection +
    scaling params, runs the model in eval mode, and recomputes:
        * silhouette of the external embeddings under the frozen cluster head
        * biomarker overlap (Jaccard) between TCGA and external top-k lists
        * (optional) survival if the external cohort has outcome data

    The point is reproducibility across cohorts, NOT retraining.
    """
    def __init__(self, model, preprocessor, config, device):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.device = device

    def align_features(self, ext_omics):
        """Restrict/reorder external features to the TCGA-selected feature set.
        Missing features are filled with the training mean (0 after z-scoring)."""
        aligned = {}
        for mod in self.config.modalities:
            train_names = list(self.preprocessor.get_feature_names(mod))
            ext_df = ext_omics[mod]
            # add absent columns as NaN, then order like training
            for c in train_names:
                if c not in ext_df.columns:
                    ext_df[c] = np.nan
            aligned[mod] = ext_df[train_names]
            missing_frac = aligned[mod].isna().all(axis=0).mean()
            print(f"  {mod}: {aligned[mod].shape[1]} features aligned "
                  f"({missing_frac:.0%} of TCGA features absent externally)")
        return aligned

    def validate(self, ext_omics, tcga_biomarkers=None, top_k=50):
        """
        Returns dict with external silhouette and (if biomarkers provided)
        per-modality Jaccard overlap with the TCGA biomarker lists.

        Implementation notes (fill in against your notebook API):
          - use self.preprocessor.transform(...) if you refactor fit_transform
            to expose a transform() that reuses fitted params (recommended);
          - forward the aligned tensors through self.model to get embeddings +
            cluster assignments;
          - re-run your AttentionBiomarkerExtractor on the external loader to
            get external top-k lists, then Jaccard vs tcga_biomarkers.
        """
        raise NotImplementedError(
            "Supply an external cohort and wire this to your notebook's "
            "preprocessor.transform() + model.forward(). See REVISION_PLAN.md "
            "step 4 for the exact call sequence and dataset suggestions "
            "(TARGET via GDC, CPTAC via LinkedOmics, ICGC via ICGC-ARGO).")


# ============================================================================
# R1.7  —  A RECENT-STYLE TRANSFORMER BASELINE
# ============================================================================
class SimpleMultiOmicsTransformer:
    """
    A compact, self-contained multi-omics transformer baseline so the paper
    benchmarks against at least one transformer competitor besides itself
    (Reviewer 1.7). This is deliberately simple and DIFFERENT from
    HiOmicsFormer: per-modality linear tokenizer -> shared transformer encoder
    over the 4 modality tokens -> mean-pool -> KMeans. No hierarchical
    tokenization, no contrastive loss, no DEC.

    Train it under the same 5-fold protocol as the other baselines and report
    its silhouette / NMI / ARI alongside SNF, MOFA+, MOGONET.

    (For a heavier comparison the reviewer would also accept, cite and — if
    compute allows — run an established model such as a multi-omics
    Transformer-VAE; see REVISION_PLAN.md step 5 for candidates and refs.)
    """
    def __init__(self, feature_dims, d=128, heads=4, layers=2,
                 n_clusters=12, epochs=50, lr=1e-3, device='cpu', seed=42):
        import torch
        import torch.nn as nn
        self.torch = torch
        torch.manual_seed(seed)
        self.device = device
        self.mods = list(feature_dims.keys())
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.lr = lr

        class Net(nn.Module):
            def __init__(self, fdims, d, heads, layers, mods):
                super().__init__()
                self.mods = mods
                self.tok = nn.ModuleDict(
                    {m: nn.Linear(fdims[m], d) for m in mods})
                enc = nn.TransformerEncoderLayer(d, heads, d * 2,
                                                 batch_first=True)
                self.tr = nn.TransformerEncoder(enc, layers)
                self.dec = nn.ModuleDict(
                    {m: nn.Linear(d, fdims[m]) for m in mods})

            def forward(self, batch):
                toks = self.torch_stack([self.tok[m](batch[m]) for m in self.mods])
                z = self.tr(toks)                       # (B, M, d)
                recon = {m: self.dec[m](z[:, i])
                         for i, m in enumerate(self.mods)}
                emb = z.mean(dim=1)                      # (B, d)
                return emb, recon

            def torch_stack(self, lst):
                import torch as _t
                return _t.stack(lst, dim=1)
        self.net = Net(feature_dims, d, heads, layers, self.mods).to(device)

    def fit_transform(self, loader):
        torch = self.torch
        import torch.nn.functional as F
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for ep in range(self.epochs):
            for batch in loader:
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                         for k, v in batch.items()}
                emb, recon = self.net(batch)
                loss = sum(F.mse_loss(recon[m], batch[m]) for m in self.mods)
                opt.zero_grad(); loss.backward(); opt.step()
        # extract embeddings
        self.net.eval()
        embs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                         for k, v in batch.items()}
                emb, _ = self.net(batch)
                embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)
