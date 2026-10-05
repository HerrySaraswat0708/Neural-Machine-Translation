#!/bin/bash
# Two limitation fixes, tested against the project winner
# (bpe_rope_mha_widernarrowffnlong, emb192/L6/ff768/drop0.1/ep20):
#  1. Joint BPE vocabulary (bpe_joint) -- shared src/tgt vocab so rare
#     tokens (names) get a consistent representation on both sides,
#     instead of relying on two independently-trained vocabs coincidentally
#     overlapping.
#  2. Static source-side noise augmentation (--noise_prob 0.5) -- typos/
#     dropped-words/local-swaps injected into a duplicated 50% slice of
#     training pairs, target unchanged, so the model sees noisy-in/clean-
#     out examples during training.
# Same architecture as the winner in both cases, so each is an isolated,
# apples-to-apples test of exactly one change.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Training: joint-vocab + noise-augmented variants ==="
CUDA_VISIBLE_DEVICES=0 python -u train.py --tokenizer bpe_joint --pe rope --attn mha --emb_dim 192 --ff_dim 768 --epochs 20                          > logs/jointvocab.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --tokenizer bpe       --pe rope --attn mha --emb_dim 192 --ff_dim 768 --epochs 20 --noise_prob 0.5 --tag noiseaug > logs/noiseaug.log   2>&1 &
wait
echo "=== Training done: $(date) ==="

echo "=== Standard clean-test-set eval (greedy) ==="
for vid in bpe_joint_rope_mha bpe_rope_mha_noiseaug; do
  python -u evaluate.py --variant "$vid" --decode greedy 2>&1 | tee -a logs/evaluate_noungrammar.log
done

echo "=== OOV proper-noun stress test (joint vocab vs. separate vocab) ==="
python -u oov_stress_test.py 2>&1 | tee logs/oov_stress_test.log

echo "=== Noise-robustness comparison (augmented vs. non-augmented) ==="
python -u noisy_robustness_eval.py 2>&1 | tee logs/noisy_robustness.log

echo "=== NOUN/GRAMMAR FIX DONE at $(date) ==="
