import time
import re
import pickle
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import jieba
from pandarallel import pandarallel


stopwords_list_path = [
    "../stopwords/baidu_stopwords.txt",
    "../stopwords/cn_stopwords.txt",
    "../stopwords/hit_stopwords.txt",
    "../stopwords/scu_stopwords.txt",
]

# tqdm.pandas()
pandarallel.initialize(progress_bar=True)


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
        if star <= 3:
            label = 0
        # elif star == 3:
        #     label = 1
        else:
            label = 1

        return self.Comments[idx], label
    

def read_comment_csv(path:str, nrows:int=None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    return df


def concat_dataframe(df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
    combined_comment = pd.concat([df1["Comment"], df2["Comment"]])
    combined_star = pd.concat([df1["Star"], df2["Star"]])
    combined_df = pd.DataFrame({"Comment": combined_comment, "Star": combined_star}).dropna()
    del df1, df2
    return combined_df


def load_stopwords(paths:list[str]) -> set[str]:
    stopwords = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f_obj:
            stopwords += [line.strip() for line in f_obj]
            
    return set(stopwords)


stopwords = None
def tokenize_str(sentence:str, use_stopwords:bool=True) -> list[str]:
    sentence = re.sub(r'[\u3000-\u303f\uff00-\uffef]', ' ', sentence)
    seg_list = list(jieba.cut(sentence.strip(), cut_all=False))
    if use_stopwords == False:
        return seg_list
    
    global stopwords
    if stopwords is None:
        stopwords = load_stopwords(stopwords_list_path)

    filtered_words = [word for word in seg_list if word not in stopwords]
    return filtered_words


def tokenizer_(use_stopwords:bool=True):
    global stopwords
    if use_stopwords and stopwords is None:
        stopwords = load_stopwords(stopwords_list_path)

    tokenize_func = partial(tokenize_str, use_stopwords=True)
    return get_tokenizer(tokenize_func)
    

def apply_tokenizer(data:pd.Series, use_stopwords:bool) -> pd.Series:
    start_time = time.perf_counter()
    tokenizer = tokenizer_(use_stopwords)
    tokenized = data.parallel_apply(tokenizer)
    print(f"finished in {time.perf_counter() - start_time}s!")
    return tokenized


def make_vocab(data:pd.Series, min_freq:int=16) -> torchtext.vocab.Vocab:
    vocab = build_vocab_from_iterator(data, min_freq=min_freq, specials=('<unk>', '<pad>'))
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def save_vocab(vocab:torchtext.vocab.Vocab, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(path:str) -> torchtext.vocab.Vocab:
    with open(path, 'rb') as f:
        loaded_vocab = pickle.load(f)
    return loaded_vocab


def padding_one_sentence(vocab:torchtext.vocab.Vocab, input:list, target_len:int=32) -> torch.Tensor:
    indexed_sequence = torch.tensor([vocab[token] for token in input]).unsqueeze(0)
    padding_size = max(0, target_len - indexed_sequence.size(1))
    padded_sequences = F.pad(indexed_sequence, [0, padding_size], value=vocab['<pad>'])
    return padded_sequences[:, :target_len]


def padding_sentences(vocab:torchtext.vocab.Vocab, data:pd.Series, target_len:int=32) -> torch.Tensor:
    indexed_sequences = [torch.tensor([vocab[token] for token in seq]) for seq in data]
    padded_sequences = pad_sequence(indexed_sequences, batch_first=True, padding_value=vocab['<pad>'])
    padded_sequences = padded_sequences[:, :target_len]
    return padded_sequences


def get_dataloader(input:torch.Tensor, label:pd.Series, batch_size:int=128, 
                   shuffle:bool=True, num_workers:int=6, pin_memory:bool=True) -> DataLoader:
    comment_set = CommentSet(input, label)
    data_loader = DataLoader(comment_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


def split_dataset(comment_set:CommentSet, batch_size:int, percentage:float=0.9, cpu_threads:int=12) -> (DataLoader, DataLoader):
    train_size = int(len(comment_set) * percentage)
    validation_size = len(comment_set) - train_size
    # print(f"train dataset size: {train_size}, validation dataset size: {validation_size}")
    train_dataset, valid_dataset = random_split(comment_set, [train_size, validation_size])

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_threads, pin_memory=True), DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=cpu_threads, pin_memory=True)