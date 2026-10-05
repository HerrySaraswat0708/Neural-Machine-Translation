#!/bin/bash
# Phase 7e draft model: same tokenizer (bpe_joint, required -- speculative
# decoding compares draft/target token ids directly, so they must share a
# vocabulary) and same pe/attn as the champion, drastically smaller
# architecture (1.78M vs 10.85M params, ~16%) for cheap proposal
# generation.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Draft model training ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer bpe_joint --pe rope --attn mha --emb_dim 64 --num_layers 2 --ff_dim 256 --epochs 20 --tag draft > logs/draft.log 2>&1
echo "=== Draft model training done: $(date) ==="

echo "=== Evaluating draft model (sanity check on translation quality) ==="
python -u evaluate.py --variant bpe_joint_rope_mha_draft --decode greedy 2>&1 | tee logs/evaluate_draft.log

echo "=== DRAFT DONE at $(date) ==="
