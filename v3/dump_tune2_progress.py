import json
m = json.load(open("results/manifest.json"))
vids = ["bpe_rope_mha_widerlong", "bpe_rope_mha_narrowffnlong", "bpe_rope_mha_widernarrowffn",
        "bpe_rope_mha_widernarrowffnlong", "bpe_rope_mha_lowdroplong", "bpe_rope_gqa_long"]
for v in vids:
    d = m.get(v, {})
    ta = d.get("final_train_acc")
    ea = d.get("final_eval_acc")
    print(v, "params=", d.get("num_params"),
          "train_acc=", round(ta, 2) if ta is not None else None,
          "eval_acc=", round(ea, 2) if ea is not None else None,
          "bleu=", d.get("bleu"))
