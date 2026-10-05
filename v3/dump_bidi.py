import json
m = json.load(open("results/manifest.json"))
print(json.dumps(m.get("bpe_joint_rope_mha_bidi", {}), indent=2))
