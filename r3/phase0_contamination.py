"""
PHASE 0 (R3) — Independently verify or refute the MLOmics contamination claim, reading
RAW CSVs directly (no project loader, no transpose logic).

Claims under test:
  1. The Clustering_datasets/THCA archive is actually colon (COAD) data.
  2. The Classification_datasets carry synthetic padding (ZZEF*/ZZZ*) + make.names mangling.

We compute concrete statistics and save results_revision/thca_contamination_proof.json.
Verdict logic is printed; the human decision is recorded in DECISIONS.md.
"""
import os, json, csv, re
import numpy as np

BASE = "data/MLOmics/Main_Dataset"
CLU = f"{BASE}/Clustering_datasets"
CLS = f"{BASE}/Classification_datasets"

# Minimal TCGA tissue-source-site -> tissue map (colon vs thyroid), from the TCGA code tables.
COLON_TSS = {"3L","4N","4T","5M","A6","AA","AD","AU","AY","AZ","CA","CK","CM","D5","DM","F4","G4","NH","QG","QL","RU","SS","T9","WS","G4","AZ"}
THYROID_TSS = {"BJ","CE","DE","DJ","E8","EL","EM","ET","FE","FK","FY","GE","H2","IM","J8","KS","L6","MK","QD"}


def read_matrix(path, max_cols=None):
    """Read an MLOmics CSV (features as rows, samples as columns). Returns (feature_names, sample_ids, np.array or None)."""
    with open(path) as fh:
        r = csv.reader(fh)
        header = next(r)
        samples = header[1:]
        feats, rows = [], []
        for row in r:
            feats.append(row[0])
            if max_cols is None:
                rows.append(row[1:])
            else:
                rows.append(row[1:max_cols+1])
    arr = None
    try:
        arr = np.array(rows, dtype=np.float64)
    except Exception:
        arr = None
    return feats, samples, arr


def tss(bc):
    # barcode like TCGA.3L.AA1B.01  -> TSS field = '3L'
    parts = re.split(r"[.\-]", bc)
    return parts[1] if len(parts) > 1 else "?"


def main():
    proof = {}

    # ---- (1) THCA archive identity ----
    thca_mrna = f"{CLU}/THCA/Original/THCA_mRNA.csv"
    coad_mrna = f"{CLS}/GS-COAD/Original/COAD_mRNA.csv"
    tf, ts, ta = read_matrix(thca_mrna)
    cf, cs, ca = read_matrix(coad_mrna)

    # (a) TSS codes of THCA barcodes
    codes = [tss(b) for b in ts]
    ucodes = sorted(set(codes))
    n_colon = sum(c in COLON_TSS for c in codes)
    n_thyroid = sum(c in THYROID_TSS for c in codes)
    proof["thca_barcode_tss"] = {
        "unique_tss_codes": ucodes,
        "n_samples": len(ts),
        "n_with_colon_tss": n_colon,
        "n_with_thyroid_tss": n_thyroid,
        "fraction_colon": round(n_colon / len(ts), 4),
    }

    # (b) barcode overlap THCA vs GS-COAD
    setT, setC = set(ts), set(cs)
    inter = setT & setC
    proof["barcode_overlap_thca_vs_gscoad"] = {
        "n_thca": len(setT), "n_gscoad": len(setC), "n_shared": len(inter),
        "frac_shared_of_smaller": round(len(inter) / min(len(setT), len(setC)), 4),
    }

    # (c) numeric identity on shared barcodes + shared features (sample of features)
    identical = None
    if inter and ta is not None and ca is not None:
        # align: map sample->col index
        tcol = {s: i for i, s in enumerate(ts)}
        ccol = {s: i for i, s in enumerate(cs)}
        trow = {f: i for i, f in enumerate(tf)}
        crow = {f: i for i, f in enumerate(cf)}
        shared_feats = [f for f in tf if f in crow][:500]  # first 500 shared genes
        shared_bcs = sorted(inter)[:100]                   # first 100 shared samples (sorted: deterministic)
        diffs, comps = 0, 0
        maxabsdiff = 0.0
        for f in shared_feats:
            for b in shared_bcs:
                tv = ta[trow[f], tcol[b]]; cv = ca[crow[f], ccol[b]]
                comps += 1
                d = abs(tv - cv)
                maxabsdiff = max(maxabsdiff, d)
                if d > 1e-6:
                    diffs += 1
        identical = {"comparisons": comps, "n_differing": diffs,
                     "max_abs_diff": maxabsdiff, "fraction_identical": round(1 - diffs / comps, 6) if comps else None}
    proof["numeric_identity_thca_vs_gscoad_mrna"] = identical

    # (d) survival_THCA barcodes -> colon?
    surv = f"{CLU}/THCA/Original/survival_THCA.csv"
    if os.path.exists(surv):
        with open(surv) as fh:
            r = csv.reader(fh); next(r)
            sbcs = [row[0] for row in r if row]
        scodes = [tss(b) for b in sbcs]
        proof["survival_thca_tss"] = {
            "n": len(sbcs), "n_colon_tss": sum(c in COLON_TSS for c in scodes),
            "n_thyroid_tss": sum(c in THYROID_TSS for c in scodes),
            "unique_tss": sorted(set(scodes)),
        }

    # ---- (2) synthetic padding in classification archives ----
    pad = re.compile(r"^(ZZEF\d+|ZZZ\d+)$")
    padding = {}
    for gs, mod in [("GS-GBM", "mRNA"), ("GS-LGG", "mRNA"), ("GS-GBM", "Methy")]:
        p = f"{CLS}/{gs}/Original/{gs[3:]}_{mod}.csv"
        if os.path.exists(p):
            with open(p) as fh:
                names = [line.split(",", 1)[0] for i, line in enumerate(fh) if i > 0]
            padlike = [n for n in names if pad.match(n)]
            padding[f"{gs}/{mod}"] = {"total_features": len(names), "padding_features": len(padlike),
                                      "examples": padlike[:5]}
    # padding present in clustering archive?
    for c, mod in [("LUSC", "mRNA"), ("KIRC", "mRNA")]:
        p = f"{CLU}/{c}/Original/{c}_{mod}.csv"
        if os.path.exists(p):
            with open(p) as fh:
                names = [line.split(",", 1)[0] for i, line in enumerate(fh) if i > 0]
            padlike = [n for n in names if pad.match(n)]
            padding[f"{c}/{mod} (clustering)"] = {"total_features": len(names), "padding_features": len(padlike)}
    proof["synthetic_padding"] = padding

    # make.names mangling (dots) in classification vs clustering
    def dot_count(path):
        with open(path) as fh:
            names = [line.split(",", 1)[0] for i, line in enumerate(fh) if i > 0]
        return sum("." in n for n in names), len(names)
    md_gs, tot_gs = dot_count(f"{CLS}/GS-GBM/Original/GBM_mRNA.csv")
    md_cl, tot_cl = dot_count(f"{CLU}/LUSC/Original/LUSC_mRNA.csv")
    proof["dot_mangling"] = {"GS-GBM_mRNA_dotted": md_gs, "GS-GBM_total": tot_gs,
                             "LUSC_clustering_dotted": md_cl, "LUSC_total": tot_cl}

    # ---- verdict ----
    # Mislabel+duplication is proven by tissue-source-site codes + barcode overlap; it does NOT
    # require the two archives' processed VALUES to be identical (they are processed differently).
    thca_is_colon = (proof["thca_barcode_tss"]["fraction_colon"] > 0.9 and
                     proof["thca_barcode_tss"]["n_with_thyroid_tss"] == 0)
    barcodes_dup = proof["barcode_overlap_thca_vs_gscoad"]["frac_shared_of_smaller"] > 0.9
    vals_identical = bool(identical and identical["fraction_identical"] > 0.99)
    # clustering "padding" of 2 are the REAL genes ZZEF1/ZZZ3 (regex false positives), not synthetic
    proof["VERDICT"] = {
        "thca_is_mislabelled_colon": bool(thca_is_colon),
        "thca_duplicates_gscoad_patients": bool(barcodes_dup),
        "matrices_numerically_identical": vals_identical,
        "correction_to_r2_claim": ("R2 said the THCA omics/survival were 'identical' to GS-COAD; raw "
                                   "files show the SAME PATIENTS (100% barcode overlap, all colon TSS) but "
                                   "NON-identical processed values across archives. The manuscript must say "
                                   "'same patients, different processing', not 'identical matrices'."),
        "padding_confirmed_in_classification": all(v.get("padding_features", 0) > 100 for k, v in padding.items() if "clustering" not in k),
        "synthetic_padding_in_clustering": any(v.get("padding_features", 0) > 5 for k, v in padding.items() if "clustering" in k),
        "basis": "raw CSVs read directly, no project loader/transpose",
    }

    os.makedirs("results_revision", exist_ok=True)
    json.dump(proof, open("results_revision/thca_contamination_proof.json", "w"), indent=2, default=str)
    print(json.dumps(proof["thca_barcode_tss"], indent=1))
    print("barcode overlap:", proof["barcode_overlap_thca_vs_gscoad"])
    print("numeric identity:", identical)
    print("survival THCA tss:", proof.get("survival_thca_tss"))
    print("padding:", {k: v.get("padding_features") for k, v in padding.items()})
    print("VERDICT:", proof["VERDICT"])


if __name__ == "__main__":
    main()
