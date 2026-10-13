# Revision analyses (R4) — reproduction guide

Everything in the revised manuscript ("HiOmicsFormer revisited: normalisation artefacts in a public
multi-omics benchmark explain apparent pan-cancer subtypes") is produced by the scripts in `r3/` and
`r4/`, from a pinned data release. Every number quoted in the text is read from the result files by
`r4/build_submission.py`; none is typed by hand.

## 1. Data (pinned)

MLOmics, Hugging Face dataset `AIBIC/MLOmics`, commit `d7c2239504dcf817ee9f33a693f9617d614eef67`
(25 May 2025). Only the `Original` folders are used:

```
Main_Dataset/Clustering_datasets/*/Original/*
Main_Dataset/Classification_datasets/GS-*/Original/*
```

Download them into `data/MLOmics/` (70 files, 4.15 GB), then verify them against the manifest:

```bash
sh r4/verify_sha256.sh r4/mlomics_manifest_d7c2239.tsv data/MLOmics    # expect: 70 ok, 0 mismatched
```

Note: the maintainers replaced the `THCA` folder (now genuine thyroid data) on 22 Sep 2026 and
re-uploaded other folders on 21 Jul 2026. Later releases will NOT reproduce these results; use the
commit above.

External annotations (fetch with `sh r4/fetch_external.sh`, which checks the SHA-256 of the versions
used, downloaded 13 Oct 2026; they are not redistributed here):
- `TCGASubtype.20170308.tsv.gz` and `Subtype_Immune_Model_Based.txt.gz` — UCSC Xena PanCanAtlas hub;
- `oncokb_cancerGeneList.txt` — OncoKB cancer gene list, including COSMIC Cancer Gene Census (v99) membership.

- MSigDB Hallmark and C6 gene sets (v2023.2.Hs) -> `data/curated/`.

## 2. Run

```bash
bash r4/run_all.sh          # PY=python3 by default; needs torch, scikit-learn, lifelines, mofapy2, statsmodels
python3 r4/tables.py        # LaTeX tables
python3 r4/fig_architecture.py
# Author side only (manuscript sources are not in the public repository):
python3 r4/build_submission.py            # assembles HiOmicsFormer_manuscript.tex (fills all numbers)
python3 r4/build_submission.py --letters  # after compiling: response letter and cover letter
```

On Apple-silicon (MPS) a full five-fold run takes about 30–50 minutes. Training is deterministic
under fixed seeds; the rerun from the pinned data reproduced the submitted cohort, features and
embeddings exactly (`results_revision/r4/reproduction_check.json`).

## 3. What each script does

| Script | Reviewer point | Output (`results_revision/…`) |
|---|---|---|
| `r3/phase0_contamination.py` | data audit | `thca_contamination_proof.json` |
| `r4/normalization_audit.py` | data audit: scale of every file; duplicated colon copies | `r4/normalization_audit.csv`, `r4/duplicate_coad_identity.json` |
| `r3/prep.py`, `r4/verify_reproduction.py` | reproduction | `r3/cache_*`, `r4/reproduction_check.json` |
| `r3/train.py --cohort all --outdir all_uncorrected --save-model` | full-cohort 5-fold | `all_uncorrected/` |
| `r3/train.py --cohort single_archive …` | single-archive 5-fold | `single_archive/` |
| `r3/dec_sweep.py` | cluster-count sweep | `dec_collapse_sweep.csv` |
| `r4/duplicate_patients.py` | R1.2 technical replicates | `r4/duplicate_patient_test.json` |
| `r4/baselines.py` | R1.2, R1.7 | `r4/baselines_{all,single_archive}.csv` |
| `r4/missingness.py` | R1.5 | `r4/missingness.json` |
| `r4/sensitivity.py` | R1.5 and the scale test | `r4/sensitivity_{all,all_fixed}.json` |
| `r4/known_subtypes.py` | R1.2 external annotations | `r4/known_subtypes.csv` |
| `r4/curated.py` | R1.4 | `r4/curated_benchmark.csv` |
| `r4/association_survival.py` | R1.1, R2 | `r4/association.json`, `r4/survival_cindex.csv` |

## 4. Corrections to earlier internal notes

- The multi-seed stability experiment of the submitted version **was run**: the original notebook,
  cell 24, printed silhouette 0.4993 ± 0.0410. It ran on the uncorrected data with the misaligned
  evaluation, so its numbers are withdrawn. Earlier internal notes that called it "fabricated" were
  wrong and have been corrected.
- The "THCA" and GS-COAD copies of the 260 colon patients are the same data, z-scored over 291
  versus 260 patients. They are not two processing pipelines (`r4/duplicate_coad_identity.json`).
- `r2/clean_loader.py`'s default loader silently drops LUAD and BRCA: their miRNA files have more
  patients than features and are read in the wrong orientation. The default is kept so that the
  analysed cohort reproduces; `OrientationFixedLoader` corrects it (used in `sensitivity.py --cohort all_fixed`).
