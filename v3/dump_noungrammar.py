import json
m = json.load(open("results/manifest.json"))
for v in ["bpe_joint_rope_mha", "bpe_rope_mha_noiseaug"]:
    print("===", v, "===")
    print(json.dumps(m.get(v, {}), indent=None))
