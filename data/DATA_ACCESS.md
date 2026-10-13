# Data access

## MLOmics (TCGA multi-omics)

- Paper: Yang Z, Kotoge R, Piao X, Chen Z, Zhu L, Gao P, Matsubara Y, Sakurai Y, Sun J.
  MLOmics: Cancer multi-omics database for machine learning. *Sci Data* 2025;12:913.
  https://doi.org/10.1038/s41597-025-05235-x
- Hugging Face: https://huggingface.co/datasets/AIBIC/MLOmics
- figshare: https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127

**Use the pinned release.** All results were produced from commit
`d7c2239504dcf817ee9f33a693f9617d614eef67` (25 May 2025). Later releases differ; for example, the
`THCA` folder was replaced with genuine thyroid data on 22 Sep 2026.

Only the `Original` folders are used:

```
data/MLOmics/Main_Dataset/Clustering_datasets/{ACC,KIRC,KIRP,LIHC,LUAD,LUSC,PRAD,THCA,THYM}/Original/*
data/MLOmics/Main_Dataset/Classification_datasets/GS-{BRCA,COAD,GBM,LGG,OV}/Original/*
```

Each file can be downloaded with

```bash
curl -L -o "data/MLOmics/<path>" \
  "https://huggingface.co/datasets/AIBIC/MLOmics/resolve/d7c2239504dcf817ee9f33a693f9617d614eef67/<path>"
```

`<path>` is a path from `r4/mlomics_manifest_d7c2239.tsv` (first column). Then verify:

```bash
sh r4/verify_sha256.sh r4/mlomics_manifest_d7c2239.tsv data/MLOmics   # expect: 70 ok
```

## Analysed cohort

After curation the analysed cohort has 3,044 patients from 11 cancer types: ACC, COAD (the `THCA`
folder, relabelled), GBM, KIRC, KIRP, LGG, LIHC, LUSC, OV, PRAD, THYM. The default loader skips LUAD
and BRCA because of a file-orientation error, which is disclosed in the manuscript.
`OrientationFixedLoader` in `r2/clean_loader.py` includes them; it is used for the 13-type
sensitivity analysis.

## Reference annotations

`sh r4/fetch_external.sh` downloads the following and checks their SHA-256 against the versions used
(13 Oct 2026):
- TCGA PanCanAtlas subtype calls and immune subtypes (UCSC Xena PanCanAtlas hub);
- the OncoKB cancer gene list, including COSMIC Cancer Gene Census v99 membership;
- MSigDB Hallmark and C6 gene sets (v2023.2.Hs).
