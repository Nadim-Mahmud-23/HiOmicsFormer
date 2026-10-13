"""
Phase 3 — curated gene-set benchmark on the single-archive model's prioritised features.

Honest gene-name handling:
  * harmonise dots->dashes;
  * flag Excel-date-corrupted symbols (pure-numeric names such as 43899 are the classic
    SEPT*/MARCH* spreadsheet corruption) and report their count -- we do NOT invent a mapping
    without the source year;
  * report the fraction of each list that is a mappable gene symbol vs non-coding/other.

Benchmark background = the modality's MAD-selected feature universe (not the genome). Uses
MSigDB Hallmark + C6 (fetched); COSMIC included automatically if data/curated/cosmic_cgc.tsv exists.
Saves results_revision/curated_benchmark_<mod>.csv + a mappability summary.
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "r2")); sys.path.insert(0, ROOT)
import torch, pickle
import nbdefs as nb
from nbdefs import config, device, HiOmicsFormer, MultiOmicsDataset, AttentionBiomarkerExtractor
from torch.utils.data import DataLoader
from hiomicsformer_revision_addons import benchmark_biomarkers_vs_curated

GENE_SYMBOL = re.compile(r'^[A-Z][A-Z0-9\-]{1,14}$')
EXCEL_NUM = re.compile(r'^\d+(\.\d+)?$')


def load_gmt(path):
    genes = set()
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            genes.update(g.strip().upper() for g in parts[2:] if g.strip())
    return genes


def classify(names):
    harm = [str(n).replace(".", "-") for n in names]
    mappable = [n for n in harm if GENE_SYMBOL.match(n) and not EXCEL_NUM.match(n)]
    excel = [n for n in harm if EXCEL_NUM.match(n)]
    noncoding = [n for n in harm if (n.startswith("LINC") or n.startswith("SNOR") or
                 n.startswith("ENSG") or "orf" in n.lower() or "-" in n and n not in mappable)]
    other = [n for n in harm if n not in mappable and n not in excel]
    return {"mappable_symbols": mappable, "excel_corrupted_numeric": excel,
            "n_total": len(harm), "n_mappable": len(mappable), "n_excel": len(excel),
            "n_noncoding_or_other": len(other)}


def main():
    out = "results_revision"
    cache = os.path.join(HERE, "cache_single_archive")
    d = np.load(os.path.join(cache, "processed_data.npz")); processed = {m: d[m] for m in config.modalities}
    meta = pickle.load(open(os.path.join(cache, "meta.pkl"), "rb"))
    config.num_clusters = meta["num_clusters"]
    pre = pickle.load(open(os.path.join(cache, "preprocessor.pkl"), "rb"))
    fd = {m: processed[m].shape[1] for m in config.modalities}
    model = HiOmicsFormer(fd, config).to(device)
    model.load_state_dict(torch.load("results_revision/single_archive/best_model.pt", map_location=device))
    model.eval()
    ds = MultiOmicsDataset(processed, meta["labels_encoded"], sample_ids=meta["sample_ids"])
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    ext = AttentionBiomarkerExtractor(model, pre, device)
    attn = ext.extract_attention_biomarkers(loader, top_k=config.top_k_biomarkers)
    recon = ext.extract_reconstruction_biomarkers(loader, top_k=config.top_k_biomarkers)

    curated = {}
    if os.path.exists("data/curated/hallmark.gmt"): curated["MSigDB_Hallmark"] = load_gmt("data/curated/hallmark.gmt")
    if os.path.exists("data/curated/c6_oncogenic.gmt"): curated["MSigDB_C6"] = load_gmt("data/curated/c6_oncogenic.gmt")
    for f in glob.glob("data/curated/*cosmic*") + glob.glob("data/curated/*census*"):
        try:
            df = pd.read_csv(f, sep="\t" if f.endswith("tsv") else ",")
            col = next((c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()), df.columns[0])
            curated["COSMIC_CGC"] = set(df[col].astype(str).str.upper())
        except Exception:
            pass

    summary = {"curated_sets": {k: len(v) for k, v in curated.items()}, "modalities": {}}
    for mod in ["mRNA", "Methy", "CNV"]:  # gene-level modalities
        attn_names = [b["name"] for b in attn.get(mod, [])]
        recon_names = [b["name"] for b in recon.get(mod, [])]
        consensus = sorted(set(attn_names) & set(recon_names))
        disc = consensus if consensus else attn_names
        cls = classify(disc)
        summary["modalities"][mod] = {"consensus_n": len(consensus), "used_list": "consensus" if consensus else "attention",
                                      **{k: v for k, v in cls.items() if k.startswith("n_")}}
        if curated and cls["mappable_symbols"]:
            bg = list(pre.get_feature_names(mod))
            bench = benchmark_biomarkers_vs_curated(cls["mappable_symbols"], bg, curated, label=mod)
            bench.to_csv(f"{out}/curated_benchmark_{mod}.csv", index=False)
            summary["modalities"][mod]["benchmark_min_p"] = float(bench["p_hypergeom"].min())
            summary["modalities"][mod]["benchmark_any_sig_fdr"] = bool((bench["p_hypergeom"] < 0.05).any())
    json.dump(summary, open(f"{out}/curated_benchmark_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
