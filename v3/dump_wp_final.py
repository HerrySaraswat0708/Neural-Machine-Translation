import json
m = json.load(open("results/manifest.json"))
for v in ["wordpiece_sinusoidal_mha", "wordpiece_rope_mha", "wordpiece_alibi_mha"]:
    print("===", v, "===")
    print(json.dumps(m.get(v, {}), indent=None))
