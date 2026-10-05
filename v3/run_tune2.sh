#!/bin/bash
# Tuning round 2: combine round 1's winning axes on bpe+RoPE+MHA.
# Round 1 found: epochs 10->20 alone was the single best change (+2.47 BLEU,
# same param count), wider (emb192) and narrowffn (ff512) each helped too,
# but depth (num_layers 8) hurt and didn't combine well with width
# (widedeep < wider alone) -- so this round tests whether the *training
# budget* win (more epochs) stacks with the *capacity* wins (width,
# narrower/more-efficient ffn), rather than re-testing depth.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Tuning round 2: combining epochs=20 with wider / narrowffn ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer bpe --pe rope --attn mha --emb_dim 192                    --epochs 20 --tag widerlong             > logs/tune2_widerlong.log             2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --tokenizer bpe --pe rope --attn mha --ff_dim 512                     --epochs 20 --tag narrowffnlong          > logs/tune2_narrowffnlong.log          2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --tokenizer bpe --pe rope --attn mha --emb_dim 192 --ff_dim 768                    --tag widernarrowffn      > logs/tune2_widernarrowffn.log         2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --tokenizer bpe --pe rope --attn mha --emb_dim 192 --ff_dim 768        --epochs 20 --tag widernarrowffnlong  > logs/tune2_widernarrowffnlong.log     2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u train.py --tokenizer bpe --pe rope --attn mha --dropout 0.05                   --epochs 20 --tag lowdroplong           > logs/tune2_lowdroplong.log            2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u train.py --tokenizer bpe --pe rope --attn gqa                                  --epochs 20 --tag long                  > logs/tune2_gqalong.log                2>&1 &
wait
echo "=== Tuning round 2 training done: $(date) ==="

echo "=== Evaluating the 6 round-2 tuning variants (greedy decode) ==="
for vid in bpe_rope_mha_widerlong bpe_rope_mha_narrowffnlong bpe_rope_mha_widernarrowffn bpe_rope_mha_widernarrowffnlong bpe_rope_mha_lowdroplong bpe_rope_gqa_long; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_tune2.log
done

echo "=== TUNE2 DONE at $(date) ==="
