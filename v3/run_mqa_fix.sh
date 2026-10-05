#!/bin/bash
# MQA fix: the only prior bpe_rope_mqa result (in manifest before this run)
# was trained under the pre-fix oversized config (84M params) and collapsed
# (eval_acc 7.2%, BLEU 0.0) -- not a fair comparison. This retrains it under
# the correct capacity config at both epoch budgets already used for GQA
# (bpe_rope_gqa / bpe_rope_gqa_long), so MHA vs MQA vs GQA -- the exact
# 3-way comparison v3/README.md's Phase 5 promises -- is finally apples-to-
# apples.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== MQA fix: retraining bpe_rope_mqa under correct capacity ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer bpe --pe rope --attn mqa                    > logs/mqa_fix_10ep.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --tokenizer bpe --pe rope --attn mqa --epochs 20 --tag long > logs/mqa_fix_20ep.log 2>&1 &
wait
echo "=== MQA fix training done: $(date) ==="

echo "=== Evaluating bpe_rope_mqa and bpe_rope_mqa_long (greedy decode) ==="
for vid in bpe_rope_mqa bpe_rope_mqa_long; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_mqa_fix.log
done

echo "=== MQA FIX DONE at $(date) ==="
