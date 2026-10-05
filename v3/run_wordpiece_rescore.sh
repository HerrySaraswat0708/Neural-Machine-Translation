#!/bin/bash
# Re-scores the 3 wordpiece variants after fixing WordPieceTokenizer.decode()
# (was dropping the space before terminal punctuation vs. this corpus's
# convention, inflating TER far more than BLEU/chrF). No retraining needed
# -- the trained checkpoints are untouched, only text reconstruction at
# eval time changes.
set -e
cd "$(dirname "$0")"
mkdir -p logs

for vid in wordpiece_sinusoidal_mha wordpiece_rope_mha wordpiece_alibi_mha; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_wordpiece_rescore.log
done

echo "=== WORDPIECE RESCORE DONE at $(date) ==="
