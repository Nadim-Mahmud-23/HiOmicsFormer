"""
CleanMLOmicsDataLoader — integrity fix for Reviewer 1.4 / 1.5 / 1.6.

Root cause found during Phase 0.1: the MLOmics *Classification* datasets (GS-GBM,
GS-LGG, GS-OV, ...) ship with
  (a) SYNTHETIC PADDING features named ZZEF2..ZZEF968, ZZZ4..ZZZ#### appended after
      the real genes (real genes ZZEF1 / ZZZ3 are kept; the numbered siblings are
      placeholders) — 2478 padding columns per GS-GBM / GS-LGG mRNA file; and
  (b) R-`make.names` MANGLED symbols (dots for dashes: 'A1BG.AS1' vs 'A1BG-AS1',
      'hsa.mir.X' vs 'hsa-mir-X'), which do NOT match the clustering-dataset names
      and therefore create spurious NaN columns after the pan-cancer concat
      (inflating the reported missingness).

The manuscript's Fig 6c 'ZZZ936 / ZZZ737' mRNA "biomarkers" are members of this
padding family (301 of the 2000 selected mRNA features were padding in the
contaminated run).

FIX (invents nothing):
  1. Harmonize every feature name: '.' -> '-'  (undoes make.names mangling).
  2. Restrict to the CLEAN gene universe = union of harmonized feature names present
     in the Clustering_datasets (which contain no padding). Any feature not in that
     universe (i.e. the synthetic padding) is dropped. Real genes like ZZEF1 survive
     because they ARE in the clustering universe.

Everything else (transpose, dedupe, common-sample logic, survival) is inherited
unchanged from the notebook's MLOmicsDataLoader.
"""
import os, glob, pickle
import numpy as np
import pandas as pd

import nbdefs as nb


def _harm(name):
    return str(name).replace(".", "-")


# ============================================================================
# Duplicate / mislabeled cancer-type correction (Reviewer 1.2 ground-truth integrity)
# ============================================================================
# VERIFIED data error: the Clustering_datasets folder labeled 'THCA' (thyroid) actually
# contains COLON (COAD) data — every barcode's TCGA tissue-source-site code is colorectal
# (3L, 4N, 4T, 5M, A6, AA, ...), survival_THCA.csv holds those same colon barcodes, and the
# THCA values equal GS-COAD's up to per-file z-scoring (291 vs 260 patients;
# re-standardising over the shared 260 reproduces GS-COAD to 1e-13, see r4/normalization_audit.py). So 'THCA' and 'COAD' are the same disease;
# 260/260 COAD barcodes are duplicated inside THCA. Left uncorrected this (a) double-counts 260
# patients, (b) mislabels 291 colon samples as thyroid, and (c) depresses NMI/ARI because the
# model correctly co-clusters the two colon groups that carry different labels.
#
# CORRECTION (invents nothing): relabel the mislabeled 'THCA' set to its true identity 'COAD',
# then drop duplicate (label, barcode) rows keeping the FIRST occurrence. Load order puts the
# survival-bearing THCA(clustering) rows before the survival-less COAD(classification) rows, so
# the kept colon set retains survival.
MISLABEL_RELABEL = {"THCA": "COAD"}


def find_duplicate_types(labels, sample_ids, min_overlap_frac=0.5):
    """Return list of (typeA, typeB, n_overlap, frac) for type pairs sharing barcodes."""
    from collections import defaultdict
    labels = list(labels)
    by = defaultdict(set)
    for lab, sid in zip(labels, sample_ids):
        by[lab].add(sid.split("_", 1)[1] if "_" in sid else sid)
    types = sorted(by)
    dups = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = types[i], types[j]
            ov = len(by[a] & by[b])
            if ov:
                frac = ov / min(len(by[a]), len(by[b]))
                if frac >= min_overlap_frac:
                    dups.append((a, b, ov, round(frac, 3)))
    return dups


def dedupe_and_relabel(omics, raw, survival, labels, sample_ids,
                       relabel_map=MISLABEL_RELABEL):
    """Apply the verified relabel + drop duplicate (label, barcode) rows.

    Operates by ROW POSITION (omics/raw DataFrames are positionally aligned with
    labels/sample_ids from the pan-cancer concat). Returns cleaned copies + a report.
    """
    labels = np.array([str(x) for x in labels], dtype=object)
    sample_ids = list(sample_ids)
    report = {"detected_duplicates": find_duplicate_types(labels, sample_ids),
              "relabel_map": relabel_map, "n_before": len(labels)}

    # 1. relabel mislabeled types to true identity
    for i in range(len(labels)):
        if labels[i] in relabel_map:
            new = relabel_map[labels[i]]
            bc = sample_ids[i].split("_", 1)[1] if "_" in sample_ids[i] else sample_ids[i]
            labels[i] = new
            sample_ids[i] = f"{new}_{bc}"

    # 2. drop duplicate (label, barcode) rows, keep first (survival-bearing) occurrence
    seen, keep = set(), []
    for i, sid in enumerate(sample_ids):
        bc = sid.split("_", 1)[1] if "_" in sid else sid
        key = (labels[i], bc)
        if key in seen:
            continue
        seen.add(key); keep.append(i)
    keep = np.array(keep, dtype=int)

    omics2 = {m: omics[m].iloc[keep].copy() for m in omics}
    raw2 = {m: raw[m].iloc[keep].copy() for m in raw} if raw is not None else None
    labels2 = labels[keep]
    sample_ids2 = [sample_ids[i] for i in keep]

    # 3. retag survival cancer_type per relabel map
    survival2 = survival
    if survival is not None and "cancer_type" in survival.columns:
        survival2 = survival.copy()
        survival2["cancer_type"] = survival2["cancer_type"].replace(relabel_map)

    report.update({"n_after": len(labels2), "n_dropped": int(len(keep.tolist()) and len(labels) - len(keep)),
                   "types_after": sorted(set(map(str, labels2)))})
    return omics2, raw2, survival2, labels2, sample_ids2, report


class CleanMLOmicsDataLoader(nb.MLOmicsDataLoader):
    def __init__(self, base_path, data_variant="Original", cache_path=None):
        super().__init__(base_path, data_variant=data_variant)
        self.dropped_report = {}   # (cancer, mod) -> {'padding_dropped', 'harmonized'}
        self.valid_universe = self._build_universe(cache_path)

    # ---- build clean per-modality gene universe from Clustering_datasets ----
    def _build_universe(self, cache_path):
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        uni = {}
        clu = str(self.clustering_path)
        for mod in ["mRNA", "miRNA", "Methy", "CNV"]:
            names = set()
            for f in glob.glob(os.path.join(clu, "*", self.data_variant, f"*_{mod}.csv")):
                with open(f) as fh:
                    for i, line in enumerate(fh):
                        if i == 0:
                            continue
                        names.add(_harm(line.split(",", 1)[0]))
            uni[mod] = names
            print(f"  [clean-universe] {mod}: {len(names)} real features (from Clustering_datasets)")
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(uni, fh)
        return uni

    # ---- override: harmonize names + drop padding after the parent loads a cancer ----
    def load_single_cancer(self, cancer_code, data_dir, source, modalities):
        omics_data, survival = super().load_single_cancer(
            cancer_code, data_dir, source, modalities)
        return self._clean_modalities(cancer_code, omics_data), survival

    def _clean_modalities(self, cancer_code, omics_data):
        for mod in list(omics_data.keys()):
            df = omics_data[mod]
            uni = self.valid_universe.get(mod)
            before = df.shape[1]
            # 1. harmonize
            df = df.copy()
            df.columns = [_harm(c) for c in df.columns]
            # drop dup names created by harmonization (keep first)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep="first")]
            # 2. restrict to clean universe (drops synthetic padding)
            if uni:
                keep = [c for c in df.columns if c in uni]
                dropped = before - len(keep)
                df = df[keep]
                if dropped > 0:
                    self.dropped_report[f"{cancer_code}/{mod}"] = {
                        "before": before, "after": len(keep), "dropped": dropped}
            omics_data[mod] = df
        return omics_data


class OrientationFixedLoader(CleanMLOmicsDataLoader):
    """Opt-in correction (R4) of an orientation error in the inherited loader.

    nb.MLOmicsDataLoader transposes a file when it has more rows than columns. MLOmics files store
    features as rows, so a file with MORE PATIENTS THAN FEATURES (LUAD and BRCA miRNA: 450 vs 427
    and 671 vs 368) is left in the wrong orientation, shares no sample IDs with the other
    modalities, and the whole cancer type is silently skipped. Here orientation is decided by
    where the TCGA barcodes are. The default CleanMLOmicsDataLoader is unchanged so that the
    analysed 11-type cohort reproduces exactly; this loader is used for the 13-type sensitivity
    analysis only.
    """

    def load_single_cancer(self, cancer_code, data_dir, source, modalities):
        omics_data, survival = nb.MLOmicsDataLoader.load_single_cancer(
            self, cancer_code, data_dir, source, modalities)
        for mod, df in list(omics_data.items()):
            idx_tcga = str(df.index[0]).startswith("TCGA")
            col_tcga = str(df.columns[0]).startswith("TCGA")
            if col_tcga and not idx_tcga:
                df = df.T
                df.index = df.index.astype(str).str.strip()
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep="first")]
                omics_data[mod] = df
        return self._clean_modalities(cancer_code, omics_data), survival
