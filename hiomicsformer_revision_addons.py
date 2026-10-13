"""
HiOmicsFormer — Revision R2 Add-On Module
==========================================
Drop-in functions that address the ABST-D-26-00152 reviewer comments.

These are written to plug into the existing notebook. They reuse the notebook's
objects by name: `config`, `preprocessor`, `raw_omics_data`, `omics_data`,
`fold_predictions`, `cancer_labels`, `sample_ids`, `survival_data`, `model`,
`full_dataset`, `device`, and the biomarker dicts `attn_biomarkers`,
`recon_biomarkers`.

Nothing here fabricates results. Every function reports whatever the data
actually produces. Where a claim in the manuscript is not supported by the
data, the corresponding function prints an explicit warning so you can correct
the text.

Reviewer-comment map (see REVISION_PLAN.md for the full table):
  R1.1  -> stratified_survival_analysis
  R1.2  -> cluster_validity_report
  R1.3  -> external_validation  (scaffold; needs an independent cohort)
  R1.4  -> benchmark_biomarkers_vs_curated  +  verify_feature_names
  R1.5  -> imputation_sensitivity_fixed
  R1.6  -> (documented reframing; no code needed beyond honest labels)
  R2    -> cluster_validity_report (purity/composition/enrichment)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, silhouette_score,
)


# ============================================================================
# R1.4 / R1.6  —  FEATURE-NAME INTEGRITY CHECK
# ============================================================================
def verify_feature_names(preprocessor, modalities, n_show=15):
    """
    Check whether the feature names carried through the pipeline are real
    biological identifiers (gene symbols, hsa-mir-*, cg* probes) or opaque
    placeholders (ZZZ###, feat_###, pure integers).

    This directly addresses the credibility of the mRNA consensus markers
    (ZZZ936, ZZZ737 in Fig. 6c). If a modality is flagged 'PLACEHOLDER-LIKE',
    you CANNOT claim biological interpretability or run enrichment for it until
    the real symbols are restored (re-map from the MLOmics feature dictionary).

    Returns a dict: modality -> {'status', 'placeholder_frac', 'examples'}.
    """
    import re
    report = {}
    # crude but effective heuristics
    gene_sym = re.compile(r'^[A-Z][A-Z0-9\-]{1,14}$')          # e.g. ABCC9, TP53
    mirna    = re.compile(r'^(hsa[-\.]?mir[-\.]?\w+)$', re.I)  # hsa-mir-124-3
    cpg      = re.compile(r'^cg\d{6,}$', re.I)                 # cg00000029
    placeholder = re.compile(r'^(ZZZ\d+|feat_\d+|\d+|X\d+|V\d+|col_?\d+)$', re.I)

    print("=" * 74)
    print("FEATURE-NAME INTEGRITY CHECK  (Reviewer 1.4 / 1.6)")
    print("=" * 74)
    for mod in modalities:
        try:
            names = [str(n) for n in preprocessor.get_feature_names(mod)]
        except Exception as e:
            report[mod] = {'status': f'ERROR: {e}', 'placeholder_frac': None,
                           'examples': []}
            print(f"  {mod:8s}: could not read feature names ({e})")
            continue
        if not names:
            report[mod] = {'status': 'EMPTY', 'placeholder_frac': None, 'examples': []}
            continue

        n = len(names)
        # placeholder takes priority: a name that looks like ZZZ936 is counted
        # as placeholder, never as a (coincidentally regex-matching) gene symbol
        n_ph = sum(bool(placeholder.match(x)) for x in names)
        n_bio = sum(bool(not placeholder.match(x) and
                         (gene_sym.match(x) or mirna.match(x) or cpg.match(x)))
                    for x in names)
        ph_frac = n_ph / n
        bio_frac = n_bio / n

        if ph_frac >= 0.5:
            status = "PLACEHOLDER-LIKE  <-- enrichment/interpretability NOT valid"
        elif bio_frac >= 0.5:
            status = "OK (biological identifiers)"
        else:
            status = "MIXED / UNCERTAIN  <-- inspect manually"

        report[mod] = {'status': status, 'placeholder_frac': ph_frac,
                       'bio_frac': bio_frac, 'examples': names[:n_show]}
        print(f"  {mod:8s}: {status}")
        print(f"           placeholder={ph_frac:5.1%}  biological={bio_frac:5.1%}  "
              f"(n={n})")
        print(f"           examples: {', '.join(names[:n_show])}")
    print("=" * 74)
    return report


# ============================================================================
# R1.5  —  IMPUTATION SENSITIVITY (FIXED)
# ============================================================================
def _mad_importance(matrix):
    """Median-absolute-deviation importance on a NON-standardized matrix.

    This is the fix for the Jaccard=1.0 artifact: the original code took
    np.var() AFTER StandardScaler (which forces unit variance), so every
    feature scored ~1.0 and rankings were floating-point noise. MAD on the
    imputed-but-unscaled matrix is a genuine, scale-aware dispersion measure
    and is exactly the criterion used for the main feature selection, so the
    sensitivity test now probes the SAME quantity the pipeline relies on.
    """
    med = np.median(matrix, axis=0)
    return np.median(np.abs(matrix - med), axis=0)


def imputation_sensitivity_fixed(raw_omics_data, config, top_k=50,
                                 seed=42, extractor_fn=None):
    """
    Corrected imputation-sensitivity analysis (Reviewer 1.5 + the Jaccard=1.0
    concern).

    Changes vs. the original cell:
      1. Ranking uses MAD on the imputed, *unstandardized* matrix (or, if you
         pass `extractor_fn`, your real attention-gradient extractor re-run per
         imputation — the gold standard). No more StandardScaler-then-variance.
      2. Reports BOTH the raw modality missingness AND the effective missingness
         after the >90%-feature / >95%-sample filtering, so the 70.63% vs 66.8%
         discrepancy in the manuscript is explained rather than hidden.
      3. Adds a random-permutation null: the Jaccard you'd expect by chance for
         two random top-50 sets drawn from the same feature pool. A "high"
         Jaccard is only meaningful relative to this null.

    Returns a tidy DataFrame you can save and a per-modality dict.

    NOTE: pass `extractor_fn(imputed_matrix, feature_names) -> ranked_names`
    to run your actual biomarker method under each imputation (recommended for
    the paper). If None, falls back to MAD importance.
    """
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer

    rng = np.random.default_rng(seed)
    rows = []
    per_mod = {}

    print("=" * 74)
    print("IMPUTATION SENSITIVITY (FIXED)  (Reviewer 1.5)")
    print("=" * 74)

    for mod in config.modalities:
        raw_df = raw_omics_data[mod]
        X = raw_df.values.astype(np.float32)
        feat_names = list(raw_df.columns)

        raw_missing = np.isnan(X).mean()

        # MAD-based feature selection (mirrors main pipeline)
        mad_all = np.nanmedian(np.abs(X - np.nanmedian(X, axis=0)), axis=0)
        k = min(config.feature_selection_top_k, X.shape[1])
        sel = np.argsort(np.nan_to_num(mad_all, nan=-np.inf))[-k:]
        Xs = X[:, sel]
        names_s = [feat_names[i] for i in sel]

        # drop features >90% missing and samples >95% missing
        fmiss = np.isnan(Xs).mean(axis=0)
        keepf = fmiss < 0.90
        Xs, names_s = Xs[:, keepf], [n for n, kf in zip(names_s, keepf) if kf]
        smiss = np.isnan(Xs).mean(axis=1)
        Xs = Xs[smiss < 0.95]

        eff_missing = np.isnan(Xs).mean() if Xs.size else np.nan
        print(f"\n--- {mod} ---")
        print(f"  raw missingness      : {raw_missing:6.1%}")
        print(f"  effective missingness: {eff_missing:6.1%}  "
              f"(after feature/sample filtering)  shape={Xs.shape}")

        if Xs.shape[0] < 50 or Xs.shape[1] < 10:
            print(f"  too little usable data — skipping")
            per_mod[mod] = {'raw_missing': raw_missing,
                            'eff_missing': eff_missing, 'skipped': True}
            continue

        # ---- imputations ----
        strategies = {}
        strategies['Mean'] = SimpleImputer(strategy='mean').fit_transform(Xs)
        strategies['k-NN'] = KNNImputer(n_neighbors=10).fit_transform(Xs)
        try:
            mice = IterativeImputer(max_iter=5, random_state=seed,
                                    n_nearest_features=min(20, Xs.shape[1]))
            strategies['MICE'] = mice.fit_transform(Xs)
        except Exception as e:
            print(f"  MICE failed ({e}); using mean as MICE fallback")
            strategies['MICE'] = strategies['Mean']

        for k_ in strategies:
            strategies[k_] = np.nan_to_num(strategies[k_], nan=0.0,
                                           posinf=0.0, neginf=0.0)

        # ---- rank biomarkers under each strategy (SAME criterion) ----
        rankings = {}
        for sname, mat in strategies.items():
            if extractor_fn is not None:
                ranked = extractor_fn(mat, names_s)[:top_k]
                rankings[sname] = set(ranked)
            else:
                imp = _mad_importance(mat)
                top = np.argsort(imp)[-top_k:][::-1]
                rankings[sname] = set(names_s[i] for i in top)

        # ---- expected Jaccard under random selection (null) ----
        pool = len(names_s)
        # E[|A∩B|] for two random size-t subsets of an n-pool ≈ t^2/n
        t = top_k
        exp_inter = (t * t) / pool
        exp_jacc_null = exp_inter / (2 * t - exp_inter) if (2 * t - exp_inter) > 0 else 0.0

        snames = list(rankings.keys())
        for i in range(len(snames)):
            for j in range(i + 1, len(snames)):
                a, b = snames[i], snames[j]
                inter = len(rankings[a] & rankings[b])
                union = len(rankings[a] | rankings[b])
                jac = inter / union if union else 0.0
                rows.append({'modality': mod, 'pair': f'{a} vs {b}',
                             'jaccard': jac, 'null_jaccard': exp_jacc_null,
                             'raw_missing': raw_missing,
                             'eff_missing': eff_missing})
                print(f"  Jaccard({a:5s}, {b:5s}) = {jac:5.3f}   "
                      f"(chance ~{exp_jacc_null:.3f})")

        per_mod[mod] = {'raw_missing': raw_missing, 'eff_missing': eff_missing,
                        'rankings': rankings, 'null_jaccard': exp_jacc_null,
                        'skipped': False}

    df = pd.DataFrame(rows)
    print("\nInterpretation guidance:")
    print("  * Report jaccard RELATIVE to null. A value near the null column")
    print("    means the two imputations agree no better than chance.")
    print("  * A value of exactly 1.00 across every pair is a red flag that")
    print("    the ranking criterion is degenerate — verify before publishing.")
    return df, per_mod


# ============================================================================
# R1.2 / R2  —  CLUSTER VALIDITY: PURITY, COMPOSITION, ENRICHMENT
# ============================================================================
def cluster_validity_report(cluster_labels, cancer_labels, save_prefix=None):
    """
    Quantitative evidence that clusters are (or are not) biologically
    structured (Reviewers 1.2 and 2).

    Produces:
      * cluster purity + overall weighted purity
      * cluster-vs-cancer-type composition (counts + row-normalized)
      * per-cell hypergeometric enrichment p-values (which cancer types are
        over-represented in which cluster), Benjamini-Hochberg corrected
      * global chi-square test of independence between cluster and cancer type
      * NMI / ARI / cluster entropy

    `cluster_labels` and `cancer_labels` are 1-D arrays of equal length.
    """
    cluster_labels = np.asarray(cluster_labels)
    cancer_labels = np.asarray(cancer_labels)
    clusters = sorted(pd.unique(cluster_labels))
    types = sorted(pd.unique(cancer_labels))
    N = len(cluster_labels)

    # composition matrix (rows=cluster, cols=cancer type)
    comp = pd.crosstab(pd.Series(cluster_labels, name='cluster'),
                       pd.Series(cancer_labels, name='cancer_type'))
    comp = comp.reindex(index=clusters, columns=types, fill_value=0)
    row_norm = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # purity
    dominant = comp.max(axis=1)
    cluster_sizes = comp.sum(axis=1)
    purity_per_cluster = (dominant / cluster_sizes.replace(0, np.nan)).fillna(0)
    overall_purity = dominant.sum() / N

    # cluster entropy (0 = pure, high = mixed)
    def _entropy(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    cluster_entropy = row_norm.apply(lambda r: _entropy(r.values), axis=1)

    # global chi-square
    chi2, chi_p, dof, _ = stats.chi2_contingency(comp.values)
    cramers_v = np.sqrt(chi2 / (N * (min(comp.shape) - 1)))

    # per-cell hypergeometric enrichment (over-representation)
    col_tot = comp.sum(axis=0)
    pvals, cells = [], []
    for ci in clusters:
        for tj in types:
            k = comp.loc[ci, tj]                    # observed in cell
            n = int(cluster_sizes[ci])              # draws (cluster size)
            K = int(col_tot[tj])                    # successes in population
            # P(X >= k) hypergeometric
            p = stats.hypergeom.sf(k - 1, N, K, n) if n > 0 and K > 0 else 1.0
            pvals.append(p)
            cells.append((ci, tj))
    # BH correction
    order = np.argsort(pvals)
    m = len(pvals)
    bh = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = m - rank
        prev = min(prev, pvals[idx] * m / r)
        bh[idx] = prev
    enrich = pd.DataFrame({'cluster': [c for c, _ in cells],
                           'cancer_type': [t for _, t in cells],
                           'p': pvals, 'fdr': bh})
    enrich_sig = enrich[enrich['fdr'] < 0.05].sort_values('fdr')

    nmi = normalized_mutual_info_score(cancer_labels, cluster_labels)
    ari = adjusted_rand_score(cancer_labels, cluster_labels)

    print("=" * 74)
    print("CLUSTER VALIDITY REPORT  (Reviewers 1.2 & 2)")
    print("=" * 74)
    print(f"  Overall weighted purity : {overall_purity:.3f}")
    print(f"  NMI (vs cancer type)    : {nmi:.3f}")
    print(f"  ARI (vs cancer type)    : {ari:.3f}")
    print(f"  Chi-square independence : chi2={chi2:.1f}, dof={dof}, "
          f"p={chi_p:.2e}, Cramer's V={cramers_v:.3f}")
    print(f"  Significant cluster<->type enrichments (FDR<0.05): "
          f"{len(enrich_sig)}")
    print("\n  Per-cluster purity / entropy / dominant type:")
    for ci in clusters:
        dom_type = comp.loc[ci].idxmax()
        print(f"    cluster {str(ci):>3s}: n={int(cluster_sizes[ci]):4d}  "
              f"purity={purity_per_cluster[ci]:.2f}  "
              f"entropy={cluster_entropy[ci]:.2f}  dominant={dom_type}")
    print("\n  Interpretation:")
    if chi_p < 0.05 and cramers_v > 0.15:
        print("    Clusters are NOT independent of cancer type (Cramer's V "
              f"= {cramers_v:.2f}). There IS biological/technical structure")
        print("    tied to tissue of origin, even if NMI is low. Report this")
        print("    — it directly answers the 'are clusters meaningful?' concern.")
    else:
        print("    Clusters show little association with cancer type. If you")
        print("    claim novel cross-tissue subtypes, back it with the")
        print("    external validation + curated-gene overlap below.")

    out = {'composition': comp, 'composition_rownorm': row_norm,
           'purity_per_cluster': purity_per_cluster,
           'overall_purity': overall_purity,
           'cluster_entropy': cluster_entropy,
           'chi2': chi2, 'chi_p': chi_p, 'cramers_v': cramers_v,
           'enrichment': enrich, 'enrichment_sig': enrich_sig,
           'nmi': nmi, 'ari': ari}
    if save_prefix:
        comp.to_csv(f"{save_prefix}_composition.csv")
        enrich.to_csv(f"{save_prefix}_type_enrichment.csv", index=False)
        print(f"\n  Saved: {save_prefix}_composition.csv, "
              f"{save_prefix}_type_enrichment.csv")
    return out


# ============================================================================
# R1.4  —  BENCHMARK BIOMARKERS AGAINST CURATED CANCER-GENE SETS
# ============================================================================
def benchmark_biomarkers_vs_curated(discovered_genes, background_genes,
                                    curated_sets, label="biomarkers"):
    """
    Test whether the discovered gene list overlaps curated cancer-gene
    resources more than expected by chance (Reviewer 1.4: "compare against
    COSMIC, MSigDB; systematic benchmarking").

    Args
    ----
    discovered_genes : iterable of gene symbols your model selected
    background_genes : iterable of ALL genes eligible for selection
                       (the post-MAD feature universe for that modality).
                       This is the correct hypergeometric background — using
                       the whole genome inflates significance.
    curated_sets     : dict name -> set/iterable of gene symbols
                       (e.g. {'COSMIC_CGC': {...}, 'MSigDB_Hallmark': {...}})
    Returns a DataFrame with overlap counts and hypergeometric p-values.
    """
    disc = set(map(str.upper, map(str, discovered_genes)))
    bg = set(map(str.upper, map(str, background_genes))) | disc
    N = len(bg)
    n = len(disc)
    rows = []
    for name, genes in curated_sets.items():
        S = set(map(str.upper, map(str, genes))) & bg   # restrict to background
        K = len(S)
        overlap = disc & S
        k = len(overlap)
        # P(X >= k)
        p = stats.hypergeom.sf(k - 1, N, K, n) if (K > 0 and n > 0) else 1.0
        exp = n * K / N if N else 0.0
        fold = (k / exp) if exp > 0 else np.nan
        rows.append({'gene_set': name, 'set_size_in_bg': K,
                     'overlap': k, 'expected': round(exp, 2),
                     'fold_enrichment': round(fold, 2) if exp > 0 else np.nan,
                     'p_hypergeom': p,
                     'overlap_genes': ';'.join(sorted(overlap))})
    df = pd.DataFrame(rows).sort_values('p_hypergeom').reset_index(drop=True)
    print("=" * 74)
    print(f"CURATED GENE-SET BENCHMARK for {label}  (Reviewer 1.4)")
    print("=" * 74)
    print(f"  discovered n={n}, background N={N}")
    for _, r in df.iterrows():
        flag = "  *" if r['p_hypergeom'] < 0.05 else ""
        print(f"  {r['gene_set']:20s} overlap={int(r['overlap']):3d}/"
              f"{int(r['set_size_in_bg']):4d}  fold={r['fold_enrichment']:>5}  "
              f"p={r['p_hypergeom']:.2e}{flag}")
    return df
