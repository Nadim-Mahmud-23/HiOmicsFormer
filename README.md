# HiOmicsFormer

Code and results for the manuscript

> **HiOmicsFormer revisited: normalisation artefacts in a public multi-omics benchmark explain apparent pan-cancer subtypes**
> Nadim Mahmud Dipu, Department of Computer Science and Engineering, Brac University (revised submission, 2026).

HiOmicsFormer is a multi-omics transformer (per-modality multi-scale transformer encoders, a
self-supervised cross-modal contrastive loss, gated cross-modal attention, a variational latent space
and a deep-embedded-clustering head) applied to mRNA, miRNA, DNA-methylation and copy-number profiles
of TCGA tumours distributed by the [MLOmics](https://huggingface.co/datasets/AIBIC/MLOmics) benchmark.

## Status of the results

The first submission reported pan-cancer subtypes and candidate biomarkers. A re-evaluation, carried
out for the revision and fully reproducible from this repository, showed that those results do not
hold:

- **Defects in the analysed MLOmics files.**
  - The folder labelled thyroid (THCA) contains colon data duplicated from GS-COAD.
  - Some classification-archive files contain placeholder feature columns and mangled gene symbols.
  - The files do not share a scale: 11 of 14 folders are z-scored within each file, whereas the GBM, LGG and OV files are not.
- **Errors in the original analysis.**
  - The clustering evaluation misaligned labels and predictions.
  - The data loader silently skipped two cancer types.
  - The written method description did not match the code (the model is unsupervised).
- **The re-evaluation.**
  - The model's apparent pan-cancer clusters separate the three differently scaled files from all others.
  - Within cancer types, the learned representation matches established TCGA subtypes less well than principal components of its input.
  - It shows no curated-gene enrichment and no more prognostic information than its input.

The notebook in `notebooks/` and the figures in `figures/` are those of the **original submission** and
are kept for provenance only; their numbers are superseded.

## Repository layout

| Path | Content |
|---|---|
| `r2/nbdefs.py` | model, training, baselines and feature scores — the code of the original notebook, extracted verbatim |
| `r2/clean_loader.py` | data loading and curation (placeholder removal, name harmonisation, relabelling and de-duplication; opt-in corrected loader) |
| `r3/` | cohort preparation (`prep.py`), five-fold training (`train.py`), cluster-count sweep, data-defect proof |
| `r4/` | revision analyses, tables and figures; `r4/README.md` maps each script to its output and to the reviewers' points |
| `results_revision/` | result files behind every number in the manuscript (`r4/`, `all_uncorrected/`, `single_archive/`; older files kept for provenance) |
| `notebooks/`, `figures/` | original submission (superseded) |

## Reproducing the results

1. **Data.** Download MLOmics at the pinned commit `d7c2239504dcf817ee9f33a693f9617d614eef67`.
   Only the `Original` folders of `Main_Dataset/Clustering_datasets/*` and
   `Main_Dataset/Classification_datasets/GS-*` are used (70 files, 4.15 GB). Place them under
   `data/MLOmics/` and verify them:
   ```bash
   sh r4/verify_sha256.sh r4/mlomics_manifest_d7c2239.tsv data/MLOmics
   ```
   See `data/DATA_ACCESS.md`. Later MLOmics releases (the THCA folder was replaced on 22 Sep 2026)
   will not reproduce these results.
2. **Reference annotations** (PanCanAtlas subtypes, OncoKB/COSMIC gene list, MSigDB Hallmark and C6),
   which are not redistributed here:
   ```bash
   sh r4/fetch_external.sh
   ```
3. **Environment.** `pip install -r requirements.txt` (Python 3.14; Apple-silicon MPS or CPU).
4. **Run everything:**
   ```bash
   bash r4/run_all.sh
   ```
   Training is deterministic under fixed seeds. A five-fold run takes about 30–50 minutes on Apple silicon.

The two trained models (`results_revision/all_uncorrected/best_model.pt`,
`results_revision/single_archive/best_model.pt`, 206 MB each) are attached to the repository's
GitHub release.

## Citation

See `CITATION.cff`. Please also cite MLOmics (Yang et al., *Scientific Data* 12:913, 2025) and the
TCGA Research Network.

## License

MIT (see `LICENSE`). Third-party data remain under their providers' terms.
