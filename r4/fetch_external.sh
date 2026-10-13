#!/usr/bin/env sh
# Download the third-party reference files used by r4/known_subtypes.py and r4/curated.py, and check
# them against the SHA-256 of the versions used in the paper (downloaded 13 Oct 2026). These files are
# not redistributed in this repository; please respect each provider's terms of use.
# Usage: sh r4/fetch_external.sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/external data/curated
fetch() {  # url  destination  expected-sha256
  curl -s -L --retry 3 -o "$2" "$1"
  got=$(shasum -a 256 "$2" | cut -d' ' -f1)
  if [ "$got" = "$3" ]; then echo "ok        $2"; else echo "CHANGED   $2 (provider has updated this file; results may differ slightly)"; fi
}
XENA=https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download
fetch "$XENA/TCGASubtype.20170308.tsv.gz"        data/external/TCGASubtype.20170308.tsv.gz        fa8f77e8b1075805befb6b39c0196b66c588c97fcb057d8ab1a2e321c6041975
fetch "$XENA/Subtype_Immune_Model_Based.txt.gz"  data/external/Subtype_Immune_Model_Based.txt.gz  d8c6da1fd2f73ed1c5c0639c66c3928679026dcacd85b52edf3265c65eff6aa5
fetch "https://www.oncokb.org/api/v1/utils/cancerGeneList.txt" data/external/oncokb_cancerGeneList.txt 576d3955b55989c9ab62c21c09a2405c4d997891c8acd7e62e6ad6272bf7b5c3
MSIG=https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs
fetch "$MSIG/h.all.v2023.2.Hs.symbols.gmt"  data/curated/hallmark.gmt      8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780
fetch "$MSIG/c6.all.v2023.2.Hs.symbols.gmt" data/curated/c6_oncogenic.gmt  7563215128be4d258105fbac7da8de4719ea7a9b3fa3e7341d3a0dc0dbc5ab98
