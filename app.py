import torch
from model import load_model
import streamlit as st
from utils import  greedy_translate

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

MAX_LENGTH = 16

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {
            "PAD": PAD_token,
            "SOS": SOS_token,
            "EOS": EOS_token,
            "UNK": UNK_token
        }
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
            UNK_token: "UNK"
        }
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.strip().split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


input_lang = Lang("eng")
output_lang = Lang("fra")
input_lang = torch.load("eng_lang.pkl",map_location='cpu', weights_only=False)
output_lang = torch.load("fra_lang.pkl",map_location='cpu', weights_only=False)



def get_model(input_lang,output_lang):
    
    model = load_model("nmt_model.pth",
                       input_lang.n_words,
                       output_lang.n_words,
                       emb_dim=128,
                       num_heads=8,
                       ff_dim=2048,
                       num_layers=6,
                       max_seq_len=MAX_LENGTH,
                       dropout=0.1,
                       pad_idx=0)
    return model


model =  get_model(input_lang,output_lang)
st.set_page_config(page_title="NMT Translator", layout="centered")
st.title("Neural Machine Translation (English → French)")
sentence = st.text_area("Enter an English sentence:", "")

if st.button("Translate"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        output = greedy_translate(model, input_lang, output_lang, sentence)
        st.success(f"**Translated:** {output}")