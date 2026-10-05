import json
m = json.load(open("results/manifest.json"))
vids = ["wordpiece_sinusoidal_mha", "wordpiece_rope_mha", "wordpiece_alibi_mha",
        "bpe_alibi_mha", "word_alibi_mha", "char_alibi_mha"]
for v in vids:
    print("===", v, "===")
    print(json.dumps(m.get(v, {}), indent=None))
