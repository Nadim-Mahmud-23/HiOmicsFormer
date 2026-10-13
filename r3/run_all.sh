#!/usr/bin/env bash
# Single command to reproduce every R3 number in the manuscript (see REPRODUCE_R3.md for detail).
# Usage: bash r3/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 r3/phase0_contamination.py
python3 r3/prep.py  --cohort single_archive
python3 r3/train.py --cohort single_archive --outdir single_archive --save-model
python3 r3/prep.py  --cohort all
python3 r3/train.py --cohort all --outdir all_batch_corrected --batch-correct
python3 r3/dec_sweep.py --cohort single_archive --epochs 35
python3 r3/phase3_curated.py
python3 r3/tests/test_transform.py
python3 scripts/check_numbers.py
echo "R3 reproduction complete; artifacts in results_revision/."
