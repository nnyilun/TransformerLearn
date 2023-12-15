import math
import os
import pickle
import random
import time
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
from module import TransformerForSentimentAnalysis
from utils import accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}, cuda_device_0:{torch.cuda.get_device_name(0)}')

# tqdm.pandas()
pandarallel.initialize(progress_bar=True)


# 数据预处理
DATA_PATH = "../dataset/DMSC_2M.csv"
df = pd.read_csv(DATA_PATH, nrows=10000)
# DATA_PATH = "../dataset/DMSC_10M.csv"
# df2 = pd.read_csv(DATA_PATH)

# combined_comment = pd.concat([df1["Comment"], df2["Comment"]])
# combined_star = pd.concat([df1["Star"], df2["Star"]])

# combined_df = pd.DataFrame({"Comment": combined_comment, "Star": combined_star}).dropna()
# del df1, df2

# combined_df_unique = combined_df.drop_duplicates().reset_index(drop=True) # 删除重复行，重置索引并覆盖原索引
# print("总数:", len(combined_df_unique)) # 总数: 11494120


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
        if star <= 3:  # 1 and 2 stars are considered negative reviews
            label = 0
        # elif star == 3:  # 3 stars are considered neutral reviews
        #     label = 1
        else:  # 4 and 5 stars are considered positive reviews
            label = 2
        return self.Comments[idx], label
    
comment_set = CommentSet(padded_sequences, df["Star"])

# 创建DataLoader
data_loader = DataLoader(comment_set, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)
comment, star = next(iter(data_loader))
print(len(comment), comment.dtype)
print(len(star), star.dtype)

# 训练过程
num_epochs = 10
num_classes = 3
learning_rate = 0.0001
batch_size = 64
vocab_len = len(vocab)
src_pad_idx = vocab['<pad>']
max_len = 64
d_model = 1024
num_heads = 16
d_ff = 2048
n_layers = 8
d_k = d_model // num_heads

model = TransformerForSentimentAnalysis(vocab_len, max_len, d_model, num_heads, d_ff, n_layers, src_pad_idx, num_classes).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    results = []
    for X, y in tqdm(data_loader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = model(X)
            l = loss(output, y)
        
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += l.item()
        results.append(accuracy(output, y))
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}, Accuracy: {sum(results) / len(results)}")


torch.save(model.state_dict(), '../model/model_statedict1213_01')

