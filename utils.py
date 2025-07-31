import torch


PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3



# def load_tokenizer(path,lang):
#     lan = Lang(lang)
#     lan =  torch.load(path, weights_only=False)
#     return lan

def sentence_to_tensor(lang, sentence):
    indices = [lang.word2index.get(word, UNK_token) for word in sentence.lower().strip().split()]
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Shape: [1, seq_len]

def greedy_translate(model, src_lang, tgt_lang, sentence, max_len=30):
    model.eval()
    src_tensor = sentence_to_tensor(src_lang, sentence).to("cpu")
    outputs = [SOS_token]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(outputs).unsqueeze(0).to("cpu")
        with torch.no_grad():
            out = model(src_tensor, tgt_tensor)
        next_token = out.argmax(-1)[:, -1].item()
        outputs.append(next_token)
        if next_token == EOS_token:
            break

    return ' '.join([tgt_lang.index2word.get(idx, "<unk>") for idx in outputs[1:-1]])
