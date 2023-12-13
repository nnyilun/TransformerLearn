import math
import os
import pickle
import random
import time
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import jieba
from tqdm import tqdm
from pandarallel import pandarallel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}, cuda_device_0:{torch.cuda.get_device_name(0)}')

# tqdm.pandas()
pandarallel.initialize(progress_bar=True)


# 数据预处理
DATA_PATH = "./dataset/DMSC_2M.csv"
df1 = pd.read_csv(DATA_PATH)
DATA_PATH = "./dataset/DMSC_10M.csv"
df2 = pd.read_csv(DATA_PATH)

combined_comment = pd.concat([df1["Comment"], df2["Comment"]])
combined_star = pd.concat([df1["Star"], df2["Star"]])

combined_df = pd.DataFrame({"Comment": combined_comment, "Star": combined_star}).dropna()
del df1, df2

combined_df_unique = combined_df.drop_duplicates().reset_index(drop=True) # 删除重复行，重置索引并覆盖原索引
print("总数:", len(combined_df_unique)) # 总数: 11494120


# 读取停用词表
stopwords_list_path = [
    "../stopwords/baidu_stopwords.txt",
    "../stopwords/cn_stopwords.txt",
    "../stopwords/hit_stopwords.txt",
    "../stopwords/scu_stopwords.txt",
]

def load_stopwords(paths:list[str]) -> set[str]:
    stopwords = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f_obj:
            stopwords += [line.strip() for line in f_obj]
            
    return set(stopwords)

stopwords = load_stopwords(stopwords_list_path)

# 使用jieba分词，可选用停用词表进行筛选
def tokenize_str(sentence:str, use_stopwords:bool=False) -> list[str]:
    seg_list = list(jieba.cut(sentence.strip(), cut_all=False))
    if use_stopwords == False:
        return seg_list
    
    filtered_words = [word for word in seg_list if word not in stopwords]
    return filtered_words

tokenize_str = partial(tokenize_str, use_stopwords=False) # 是否启用停用词表(停用词表存在"不"等词语)
tokenizer = get_tokenizer(tokenize_str)

start_time = time.perf_counter()
# tokenized = df['Comment'].progress_apply(tokenizer)
tokenized = df['Comment'].parallel_apply(tokenizer)
print(f"finished in {time.perf_counter() - start_time}s!")

# 构建词汇表
vocab = build_vocab_from_iterator(tokenized, min_freq=2, specials=('<unk>', '<pad>'))
vocab.set_default_index(vocab["<unk>"]) # 将不在词汇表中的词语设置为"<unk>"
print(f"vocab len:{len(vocab)}") # vocab_len(47088)

filtered_df = df[df['length'] < 64]
num_sentences_less_than = filtered_df.shape[0]
print(num_sentences_less_than, f'{num_sentences_less_than / len(df) * 100}%')
# 长度小于64的评论有1938233条，占比92.21%

# 统一句子长度
indexed_sequences = [torch.tensor([vocab[token] for token in seq]) for seq in tokenized]
padded_sequences = pad_sequence(indexed_sequences, batch_first=True, padding_value=vocab['<pad>'])
target_length = 64
padded_sequences = padded_sequences[:, :target_length]


# 创建Dataset和DataLoader
class CommentSet(Dataset):
    def __init__(self, Comments:torch.Tensor, Stars:pd.Series) -> None:
        assert Stars.min() >= 1 and Stars.max() <= 5, "Label out of range"
        assert len(Comments) == len(Stars), "The number of comments and stars is not the same!"
        self.Comments = Comments.long()
        self.Stars = torch.tensor(Stars, dtype=torch.long)
        
    def __len__(self):
        return len(self.Comments)

    def __getitem__(self, idx:int) -> tuple:
        star = self.Stars[idx] - 1
        # assert 0 <= star < 5, "The converted label is out of range"
        if star <= 2:  # 1 and 2 stars are considered negative reviews
            label = 0
        elif star == 3:  # 3 stars are considered neutral reviews
            label = 1
        else:  # 4 and 5 stars are considered positive reviews
            label = 2
        return self.Comments[idx], label
    
comment_set = CommentSet(padded_sequences, df["Star"])

# 创建DataLoader
data_loader = DataLoader(comment_set, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)
comment, star = next(iter(data_loader))
print(len(comment), comment.dtype)
print(len(star), star.dtype)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, max_len:int, embed_dim:int, dropout:float=0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 分子部分，生成[0, max_len)的序列，并且添加一个维度
        _2i = torch.arange(0, embed_dim, step=2).float() # 生成_2序列
        
        # 奇数列为正弦，偶数列为余弦
        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / embed_dim)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / embed_dim)))
        
        # 将位置编码矩阵 pe 注册为一个 buffer，被认为是持久的值，不会作为模型参数，可添加到gpu中
        self.register_buffer('pe', pe)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x形状为[btach_size, sentence_len, embed_dim]，x.size(1)为句子长度，即位置编码张量的句子长度应该大于等于输入的句子长度，也是pos的范围
        # print(f'x:{x.shape}, self.pe:{self.pe.shape}')
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# 注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_k:int, dropout:float=0.1) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = dim_k
        self.sqrt_dk = torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor=None) -> (torch.Tensor, torch.Tensor):
        # q和k维度相同，所以需要将k转置一下
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dk
        
        if mask is not None: # 不直接if mask是为了避免张量全为0而跳过掩码处理的可能
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min) # torch.finfo(scores.dtype).min是dtype类型的最小值
            
        # 计算 softmax 得到注意力权重
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        
        return output, attention
 
# Mask
def create_padding_mask(seq_q:torch.Tensor, seq_k:torch.Tensor, pad_idx:int) -> torch.Tensor:
    """ 创建注意力掩码以识别序列中的<pad>占位符
    Args:
        seq_q (Tensor): query序列，shape = [batch size, query len]
        seq_k (Tensor): key序列，shape = [batch size, key len]
        pad_idx (int): key序列<pad>占位符对应的数字索引
    """
    # 创建一个掩码，其中seq_k等于pad_idx的位置为True
    pad_attn_mask = (seq_k == pad_idx)

    # 调整掩码的形状以匹配注意力权重矩阵的形状
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(-1, seq_q.size(1), -1)

    return pad_attn_mask == False

# 多头注意力(Multi-Head Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention = ScaledDotProductAttention(self.d_k, dropout=dropout)

        # 定义线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Q K V的形状为[batch_size, seq_len, d_model]
        batch_size = query.size(0)
        seq_len = query.size(1)

        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算多头注意力
        mask = mask.unsqueeze(1)  # 添加一个维度
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)  # 扩展mask以匹配多头注意力
        output, attention = self.attention(query, key, value, mask)

        # 拼接头部并应用最终线性层
        # contiguous()类似深拷贝，transpose并不会生成新的张量，修改会修改到原变量
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention

# 基于位置的前馈神经网络（Position-Wise Feed-Forward Network）
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.FFN(X)

# Add & Norm
class AddNorm(nn.Module):
    def __init__(self, size:int, dropout:float=0.1) -> None:
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor, sublayer:torch.Tensor) -> torch.Tensor:
        # print(f'x:{x.shape}, sublayer:{sublayer.shape}')
        return self.norm(self.dropout(x) + sublayer)

# Encoder Layer
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
        """
        encoder_inputs: [batch_size, src_len, d_model]
        encoder_attention_mask: [batch_size, src_len, src_len]
        """
        residual = encoder_inputs
        # print(f'residual:{residual.shape}')
        enc_outputs, attn = self.attention(encoder_inputs, encoder_inputs, encoder_inputs, encoder_attention_mask)
        # print(f'enc_outputs:{enc_outputs.shape}, residual:{residual.shape}')
        enc_outputs = self.add_norm1(enc_outputs, residual)
        
        residual = enc_outputs
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = self.add_norm2(enc_outputs, residual)

        return enc_outputs, attn

# 编码器
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


# 情感分析任务
class TransformerForSentimentAnalysis(nn.Module):
    def __init__(self, 
                 src_vocab_size:int, max_len:int, d_model:int, 
                 n_heads:int, d_ff:int, n_layers:int, 
                 src_pad_idx:int, num_classes:int, dropout_p=0.1) -> None:
        super(TransformerForSentimentAnalysis, self).__init__()
        self.encoder = Encoder(src_vocab_size, max_len, d_model, n_heads, d_ff, n_layers, src_pad_idx, dropout_p)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src) -> torch.Tensor:
        # 通过编码器获取编码序列
        enc_outputs, _ = self.encoder(src)

        # 使用编码序列的第一个元素（对应于起始标记）来进行分类
        # 从编码器输出的每个批次中提取全局表示，通常是序列的第一个元素
        # out维度为[batch_size, num_classes]
        out = self.fc(enc_outputs[:, 0, :])

        return out


# 训练过程
num_epochs = 10
num_classes = 3
learning_rate = 0.01
batch_size = 256
vocab_len = len(vocab)
src_pad_idx = vocab['<pad>']
max_len = 64
d_model = 512
num_heads = 8
d_ff = 2048
n_layers = 6
d_k = d_model // num_heads

model = TransformerForSentimentAnalysis(vocab_len, max_len, d_model, num_heads, d_ff, n_layers, src_pad_idx, num_classes).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X, y in tqdm(data_loader):
        X, y = X.to(device), y.to(device)
        
        output = model(X)
        l = loss(output, y)
        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        total_loss += l.item()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}")


torch.save(model.state_dict(), '../model/model_statedict1213_01')

