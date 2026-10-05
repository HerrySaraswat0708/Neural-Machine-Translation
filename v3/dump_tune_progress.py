import json
m = json.load(open("results/manifest.json"))
vids = ["bpe_rope_mha_deeper", "bpe_rope_mha_wider", "bpe_rope_mha_widedeep",
        "bpe_rope_mha_narrowffn", "bpe_rope_mha_longtrain", "bpe_rope_mha_lowdrop",
        "bpe_rope_gqa"]
for v in vids:
    d = m.get(v, {})
    ta = d.get("final_train_acc")
    ea = d.get("final_eval_acc")
    print(v, "params=", d.get("num_params"),
          "train_acc=", round(ta, 2) if ta is not None else None,
          "eval_acc=", round(ea, 2) if ea is not None else None,
          "bleu=", d.get("bleu"))
