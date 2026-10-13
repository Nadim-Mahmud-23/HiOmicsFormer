#!/bin/sh
# usage: verify_sha.sh <manifest.tsv> <data_root>
ok=0; bad=0
while IFS="$(printf '\t')" read -r p size sha; do
  f="$2/$p"
  if [ -f "$f" ] && [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$sha" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "MISMATCH $p"; fi
done < "$1"
echo "sha256 verified: $ok ok, $bad mismatched/missing"
