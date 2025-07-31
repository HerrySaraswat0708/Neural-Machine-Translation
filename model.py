from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import numpy as np




PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 16
batch_size = 32

device = 'cpu'


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention,self).__init__()
        assert emb_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.emb_dim = emb_dim

        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_0 = nn.Linear(emb_dim, emb_dim)

    def forward(self, q,k,v, mask=None):
        batch_size, seq_len, emb_dim = q.shape
        H = self.num_heads
        D = self.head_dim

        Q = self.W_q(q).reshape(batch_size, -1, H, D).transpose(1, 2)
        K = self.W_k(k).reshape(batch_size, -1, H, D).transpose(1, 2)
        V = self.W_v(v).reshape(batch_size, -1, H, D).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (D ** 0.5) 
        
        
        if mask is not None:
            scores = scores.masked_fill(mask == 1, value=-1e9)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V 
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        output_final = self.W_0(out)
        return output_final

class PositionalEncoding(nn.Module):
    def __init__(self,max_seq_len,emb_dim):
        super().__init__()
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2, dtype=torch.float) * -(np.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term) # takes vectors from 0 with step of 2 
        pe[:, 1::2] = torch.cos(position * div_term) # takes vectors from 1 with step of 2 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.1)
        self.pos_emb = PositionalEncoding(max_seq_len, emb_dim)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        return x
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class AddNorm(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer))


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(emb_dim, num_heads)
        self.addnorm1 = AddNorm(emb_dim, dropout)
        self.ffn = PositionwiseFeedForward(emb_dim, ff_dim, dropout)
        self.addnorm2 = AddNorm(emb_dim, dropout)

    def forward(self, x, enc_mask=None):
        x = self.addnorm1(x, self.mha(x,x,x,enc_mask))
        x = self.addnorm2(x, self.ffn(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(emb_dim, num_heads)
        self.addnorm1 = AddNorm(emb_dim, dropout)
        
        self.cross_mha = MultiHeadAttention(emb_dim, num_heads)
        self.addnorm2 = AddNorm(emb_dim, dropout)

        self.ffn = PositionwiseFeedForward(emb_dim, ff_dim, dropout)
        self.addnorm3 = AddNorm(emb_dim, dropout)

    def forward(self, x, enc_out, tgt_mask=None, enc_mask=None):
        # Masked self-attention
        x = self.addnorm1(x, self.masked_mha(x, x, x, tgt_mask))

        # Encoder-decoder cross attention
        x = self.addnorm2(x, self.cross_mha(x, enc_out, enc_out, enc_mask))

        # Feed-forward
        x = self.addnorm3(x, self.ffn(x))
        return x

def generate_padding_mask(src, pad_idx=0):
    return (src == pad_idx).type(torch.int16).unsqueeze(-2).unsqueeze(-2)

def generate_subsequent_mask(seq_len, device=None):
    mask = torch.triu(torch.ones((1,seq_len, seq_len)),diagonal=1).type(torch.int16)
    return mask

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=512, num_heads=8,
                 ff_dim=2048, num_layers=6, max_seq_len=256, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_embed = TransformerEmbedding(src_vocab_size, emb_dim, max_seq_len)
        self.tgt_embed = TransformerEmbedding(tgt_vocab_size, emb_dim, max_seq_len)

        self.encoder = nn.ModuleList([
            EncoderBlock(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(emb_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        src_mask = (src == self.pad_idx).unsqueeze(-2).unsqueeze(-2).to(src.device)  
        tgt_mask = generate_subsequent_mask(tgt.shape[1]).to(tgt.device)     
        tgt_pad_mask = (tgt == self.pad_idx).unsqueeze(-2).to(tgt.device)  
        combined_tgt_mask = tgt_pad_mask | tgt_mask 
        combined_tgt_mask = combined_tgt_mask.unsqueeze(1)
    
        # Embedding
        src_emb = self.src_embed(src)  
        tgt_emb = self.tgt_embed(tgt)  
    
        # Encoder
        enc_out = src_emb
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask)
    
        # Decoder
        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, combined_tgt_mask, src_mask)
    
        # Final projection to vocabulary size
        output = self.output_proj(dec_out)
        return output


def load_model(path, src_vocab_size, tgt_vocab_size, emb_dim, num_heads, ff_dim, num_layers, max_seq_len, dropout, pad_idx):
    model = Transformer(src_vocab_size, tgt_vocab_size, emb_dim, num_heads, ff_dim, num_layers, max_seq_len, dropout, pad_idx)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model



