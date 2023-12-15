import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, max_len:int, embed_dim:int, dropout:float=0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, embed_dim, step=2).float()

        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / embed_dim)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / embed_dim)))

        self.register_buffer('pe', pe)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_k:int, dropout:float=0.1) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = dim_k
        self.sqrt_dk = torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor=None) -> (torch.Tensor, torch.Tensor):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dk
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attention = self.softmax(scores)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        
        return output, attention
 

def create_padding_mask(seq_q:torch.Tensor, seq_k:torch.Tensor, pad_idx:int) -> torch.Tensor:
    pad_attn_mask = (seq_k == pad_idx)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(-1, seq_q.size(1), -1)

    return pad_attn_mask == False


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention = ScaledDotProductAttention(self.d_k, dropout=dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor) -> (torch.Tensor, torch.Tensor):
        batch_size = query.size(0)
        seq_len = query.size(1)

        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)


        mask = mask.unsqueeze(1) 
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        output, attention = self.attention(query, key, value, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.FFN(X)


class AddNorm(nn.Module):
    def __init__(self, size:int, dropout:float=0.1) -> None:
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor, sublayer:torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(x) + sublayer)


class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.d_k = d_model // n_heads
        
        if self.d_k * n_heads != d_model:
            raise ValueError(f"`d_model` {d_model} can not be divisible by `num_heads` {n_heads}!")
            
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        
    def forward(self, encoder_inputs:torch.Tensor, encoder_attention_mask:torch.Tensor) -> (torch.Tensor, torch.Tensor):
        residual = encoder_inputs
        enc_outputs, attn = self.attention(encoder_inputs, encoder_inputs, encoder_inputs, encoder_attention_mask)
        enc_outputs = self.add_norm1(enc_outputs, residual)
        
        residual = enc_outputs
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = self.add_norm2(enc_outputs, residual)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size:int, max_len:int, d_model:int, n_heads:int, d_ff:int, n_layers:int, src_pad_idx:int, dropout_p:float=0.1):
        super(Encoder, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_ffn = PositionalEncoding(max_len, d_model, dropout_p)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_p) for _ in range(n_layers)])
        self.scaling_factor = math.sqrt(d_model)

    def forward(self, enc_inputs:torch.Tensor) -> torch.Tensor:
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs * self.scaling_factor)

        enc_self_attn_mask = create_padding_mask(enc_inputs, enc_inputs, self.src_pad_idx)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns 


class TransformerForSentimentAnalysis(nn.Module):
    def __init__(self, 
                 src_vocab_size:int, max_len:int, d_model:int, 
                 n_heads:int, d_ff:int, n_layers:int, 
                 src_pad_idx:int, num_classes:int, dropout_p=0.1) -> None:
        super(TransformerForSentimentAnalysis, self).__init__()
        self.encoder = Encoder(src_vocab_size, max_len, d_model, n_heads, d_ff, n_layers, src_pad_idx, dropout_p)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src) -> torch.Tensor:
        enc_outputs, _ = self.encoder(src)
        out = self.fc(enc_outputs[:, 0, :])

        return out


class SentimentModel(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, num_heads:int, d_ff:int, num_encoder_layers:int, num_classes:int, activation:torch.nn.functional=F.gelu, dropout:float=0.1):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=d_ff, activation=activation, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        out = transformer_output.mean(dim=1)
        return self.fc(out)