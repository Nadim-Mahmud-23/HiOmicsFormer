#!/usr/bin/env bash
# Reproduce every revision (R4) analysis, table and figure from the pinned data releases.
#   Data: Hugging Face AIBIC/MLOmics, commit d7c2239504dcf817ee9f33a693f9617d614eef67,
#         Main_Dataset/{Clustering_datasets/*,Classification_datasets/GS-*}/Original/*   -> data/MLOmics/
#         verify: sh r4/verify_sha256.sh r4/mlomics_manifest_d7c2239.tsv data/MLOmics
#   External annotations: sh r4/fetch_external.sh  (-> data/external/, data/curated/)
# Usage: bash r4/run_all.sh        (Python with torch, scikit-learn, lifelines, mofapy2, statsmodels, python-docx)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
# --- data audit ---------------------------------------------------------------------------
$PY r3/phase0_contamination.py                                  # mislabelled/duplicated folder, placeholders
$PY r4/normalization_audit.py                                   # file scales, duplicate identity, converted symbols
# --- cohorts and reproduction ---------------------------------------------------------------
$PY r2/phase0_setup.py                                          # imputation-ranking sensitivity (Supp. Fig. S3)
$PY r3/prep.py --cohort single_archive                          # caches (curation, MAD, kNN, scaling)
$PY r3/prep.py --cohort all
$PY r4/verify_reproduction.py                                   # caches identical to the submitted analysis
# --- models ---------------------------------------------------------------------------------
$PY r3/train.py --cohort all --outdir all_uncorrected --save-model
$PY r3/train.py --cohort single_archive --outdir single_archive --save-model
$PY r3/dec_sweep.py --cohort single_archive --epochs 35
# --- analyses -------------------------------------------------------------------------------
$PY r4/duplicate_patients.py
$PY r4/baselines.py --cohort single_archive
$PY r4/baselines.py --cohort all
$PY r4/missingness.py
$PY r4/sensitivity.py --cohort all
$PY r4/sensitivity.py --cohort all_fixed --variants reference per_type_standardized
$PY r4/scale_signature.py
$PY r4/known_subtypes.py
$PY r4/curated.py
$PY r4/association_survival.py
# --- outputs --------------------------------------------------------------------------------
$PY r4/fig_architecture.py
$PY r4/figures.py
$PY r4/tables.py
if [ -d r4/manuscript_parts ]; then   # manuscript sources are kept with the author, not in the public repository
  $PY r4/build_submission.py
  echo "Now compile HiOmicsFormer_Elsevier_Submission/{HiOmicsFormer_manuscript,Supplementary}.tex (pdflatex x2),"
  echo "then: $PY r4/build_submission.py --letters && $PY r4/package_submission.py"
fi
echo "Done: results in results_revision/, figures and tables in HiOmicsFormer_Elsevier_Submission/."
