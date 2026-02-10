# Notebooks

This directory contains the main analysis notebooks for HiOmicsFormer.

## Main Notebook

**HiOmicsFormer_Publication_Ready_Updated.ipynb**

This is the primary notebook containing the complete analysis pipeline for the manuscript submitted to *Advances in Biomarker Sciences and Technology*.

### Notebook Structure

The notebook is organized into five main parts:

1. **Part I: Introduction & Data**
   - Environment setup
   - Data acquisition from MLOmics dataset
   - Comprehensive exploratory data analysis

2. **Part II: Methodology**
   - Data preprocessing and feature engineering
   - HiOmicsFormer architecture implementation
   - Training framework with contrastive learning

3. **Part III: Results**
   - Model evaluation and comparison with baselines
   - Ablation studies
   - Subtype discovery and characterization
   - Biomarker extraction and ranking

4. **Part IV: Validation**
   - Survival analysis with Kaplan-Meier curves
   - Functional enrichment analysis
   - Literature validation
   - Cross-validation and robustness testing

5. **Part V: Discussion & Export**
   - Results summary and interpretation
   - Publication-ready figures and tables
   - Data export for supplementary materials

### Running the Notebook

```bash
# From the project root directory
jupyter notebook notebooks/HiOmicsFormer_Publication_Ready_Updated.ipynb
```

### Important Notes

- **First cell**: Always run the first cell which imports configuration from `src/config.py`
- **Runtime**: Full analysis takes approximately 2-3 hours on Apple M4 Max
- **Memory**: Requires at least 16GB RAM (32GB recommended)
- **Data**: Ensure MLOmics dataset is downloaded and path is configured in `src/config.py`

### Output

The notebook generates:
- Model checkpoints in `checkpoints/`
- Results files in `results/`
- Figures in `figures/main/` and `figures/supplementary/`
- Processed data in `processed_data/`
- Logs in `logs/`
