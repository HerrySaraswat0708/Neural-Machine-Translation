#!/bin/bash
# Bidirectional en<->fr in ONE model (Johnson et al. 2017's target-tag
# trick): source tagged "tofr "/"toen " tells the decoder which language
# to produce. Requires the joint vocab (bpe_joint) since one model now
# serves both languages on both the encoder and decoder side. Same
# winning architecture (emb192/L6/ff768) as the champion, trained on the
# doubled (213,062-pair) bidirectional dataset.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Bidirectional training: en<->fr, one model ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer bpe_joint --pe rope --attn mha --emb_dim 192 --ff_dim 768 --epochs 20 --bidirectional --tag bidi > logs/bidi.log 2>&1
echo "=== Bidirectional training done: $(date) ==="

echo "=== Evaluating both directions separately ==="
CUDA_VISIBLE_DEVICES=0 python -u evaluate_bidirectional.py --variant bpe_joint_rope_mha_bidi 2>&1 | tee logs/evaluate_bidi.log

echo "=== BIDI DONE at $(date) ==="
