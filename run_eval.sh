#!/bin/bash
set -e
cd /home/herrys/projects/Neural-Machine-Translation
LOG=run_all.log

echo "" | tee -a $LOG
echo "=== Eval restarted from bpe_rope at $(date) ===" | tee -a $LOG

# Greedy pass for remaining variants
for variant in bpe_rope char_sinusoidal char_rope word_rope; do
    echo ">>> greedy: $variant" | tee -a $LOG
    python -u evaluate.py --variant $variant --decode greedy 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== All greedy done. Running all-decodes for all 6 variants ===" | tee -a $LOG
python -u evaluate.py --all --all-decodes 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Evaluation complete at $(date) ===" | tee -a $LOG
