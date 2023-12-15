import math
import os
import json
import pickle
import random
import time
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import jieba
from tqdm import tqdm
from pandarallel import pandarallel

cpu_threads = torch.get_num_threads()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'cpu thread num: {cpu_threads}')
print(f'device is {device}, cuda_device_0:{torch.cuda.get_device_name(0)}')

# tqdm.pandas()
pandarallel.initialize(progress_bar=True, nb_workers=cpu_threads)


nrows = 100000
DATA_PATH = "../dataset/DMSC_2M.csv"
df1 = pd.read_csv(DATA_PATH, nrows=nrows)

print("总数:", len(df1))
print("每列包含的内容:", df1.columns.tolist())
print("缺失元素:", df1.isnull().sum().to_dict())


combined_df = pd.DataFrame({"Comment": df1["Comment"], "Star": df1["Star"]}).dropna()
del df1

combined_df_unique = combined_df.drop_duplicates().reset_index(drop=True) # 删除重复行，重置索引并覆盖原索引
print("总数:", len(combined_df_unique)) # 12M数据集有效评论总数: 11494120

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

def tokenize_str(sentence:str, use_stopwords:bool=False) -> list[str]:
    seg_list = list(jieba.cut(sentence.strip(), cut_all=False))
    if use_stopwords == False:
        return seg_list
    
    filtered_words = [word for word in seg_list if word not in stopwords]
    return filtered_words

tokenize_str = partial(tokenize_str, use_stopwords=False) # 是否启用停用词表
tokenizer = get_tokenizer(tokenize_str)

start_time = time.perf_counter()
# tokenized = df['Comment'].progress_apply(tokenizer)
tokenized = combined_df_unique['Comment'].parallel_apply(tokenizer)
print(f"finished in {time.perf_counter() - start_time}s!")

vocab = build_vocab_from_iterator(tokenized, min_freq=1, specials=('<unk>', '<pad>'))
vocab.set_default_index(vocab["<unk>"]) # 将不在词汇表中的词语设置为"<unk>"
print(f"vocab len:{len(vocab)}")


df_length = tokenized.apply(len)
filtered_df = tokenized[df_length < 64]
num_sentences_less_than = filtered_df.shape[0]
print(num_sentences_less_than, f'{num_sentences_less_than / len(tokenized) * 100}%')

indexed_sequences = [torch.tensor([vocab[token] for token in seq]) for seq in tokenized]
padded_sequences = pad_sequence(indexed_sequences, batch_first=True, padding_value=vocab['<pad>'])
target_length = 64
padded_sequences = padded_sequences[:, :target_length]


class CommentSet(Dataset):
    def __init__(self, Comments:pd.Series, Stars:pd.Series) -> None:
        assert Stars.min() >= 1 and Stars.max() <= 5, "Label out of range"
        assert len(Comments) == len(Stars), "The number of comments and stars is not the same!"
        self.Comments = Comments
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
    
comment_set = CommentSet(padded_sequences, combined_df_unique["Star"])

percentage = 0.9
train_size = int(len(comment_set) * percentage)
validation_size = len(comment_set) - train_size
print(f"train dataset size: {train_size}, validation dataset size: {validation_size}")
train_dataset, valid_dataset = random_split(comment_set, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=cpu_threads, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=cpu_threads, pin_memory=True)


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


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_encoder_layers, num_classes):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        out = transformer_output.mean(dim=1)
        return self.fc(out)


num_epochs = 10
num_classes = 3
learning_rate = 0.0001
batch_size = 32
vocab_len = len(vocab)
max_len = 64
d_model = 512
num_heads = 8
n_layers = 3

model = SentimentModel(vocab_len, d_model, num_heads, n_layers, num_classes).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()


def evaluate(model:torch.nn.Module, valid_data:torch.utils.data.dataloader.DataLoader) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in valid_data:
            X, y = X.to(device), y.to(device)
            
            # with autocast():
            output = model(X)
            
            predictions = torch.argmax(output, dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)

    # 计算准确率
    accuracy = total_correct / total_samples
    return accuracy


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X, y in tqdm(train_loader):
        X, y = X.to(device), y.to(device)
        
        # with autocast():
        output = model(X)
        l = loss(output, y)
        
        optimizer.zero_grad()
        # scaler.scale(l).backward()
        # scaler.step(optimizer)
        # scaler.update()
        l.backward()
        optimizer.step()
        
        total_loss += l.item()
    
    accuracy = evaluate(model, valid_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}, Validation Accuracy: {accuracy:.4f}")


torch.save(model.state_dict(), '../model/model_statedict1214_01')