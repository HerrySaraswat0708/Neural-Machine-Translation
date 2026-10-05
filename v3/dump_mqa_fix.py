import json
m = json.load(open("results/manifest.json"))
for v in ["bpe_rope_mqa", "bpe_rope_mqa_long", "bpe_rope_gqa", "bpe_rope_gqa_long", "bpe_rope_mha", "bpe_rope_mha_longtrain"]:
    print("===", v, "===")
    print(json.dumps(m.get(v, {}), indent=None))
