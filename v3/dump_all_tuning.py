import json
m = json.load(open("results/manifest.json"))
vids = [
    "bpe_rope_mha", "bpe_rope_mha_deeper", "bpe_rope_mha_wider", "bpe_rope_mha_widedeep",
    "bpe_rope_mha_narrowffn", "bpe_rope_mha_longtrain", "bpe_rope_mha_lowdrop", "bpe_rope_gqa",
    "bpe_rope_mha_widerlong", "bpe_rope_mha_narrowffnlong", "bpe_rope_mha_widernarrowffn",
    "bpe_rope_mha_widernarrowffnlong", "bpe_rope_mha_lowdroplong", "bpe_rope_gqa_long",
]
for v in vids:
    d = m.get(v, {})
    print("===", v, "===")
    print(json.dumps(d, indent=None))
