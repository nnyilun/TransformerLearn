import re
import jieba
import pandas as pd
from nltk.corpus import stopwords
from pandarallel import pandarallel
from collections import Counter
from torchtext.data.utils import get_tokenizer
from data import read_comment_csv, concat_dataframe, load_stopwords

# pandarallel.initialize(progress_bar=True)

df1 = read_comment_csv("../dataset/DMSC_2M.csv")
df2 = read_comment_csv("../dataset/DMSC_10M.csv")
combined_df = concat_dataframe(df1, df2)


my_stopwords = load_stopwords()
def tokenize_str(sentence:str) -> list[str]:
    sentence = str(sentence)
    sentence = re.sub(r'[\u3000-\u303f\uff00-\uffef]', ' ', sentence)
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, '', sentence)
    sentence = re.sub(r'http\S+', '',sentence)
    sentence = re.sub('[0-9]+', '', sentence)

    stop_words = set(stopwords.words('chinese'))
    seg_list = list(jieba.cut(sentence.strip(), cut_all=True))
    seg_list = [w for w in seg_list if not w.lower() in stop_words]
    seg_list = [word for word in seg_list if word not in my_stopwords]
    return seg_list

tokenizer = get_tokenizer(tokenize_str)
tokenized = combined_df["Comment"].parallel_apply(tokenizer)

all_words = []
for words_list in tokenized:
    all_words += words_list

word_counts = Counter(all_words)
common_words = word_counts.most_common()

words = []
for word in common_words:
    if word[1] >= 1000:
        words.append(word[0])

with open('../dataset/high_frequency_word.txt', 'w') as f:
    for w in words:
        f.write(f'{w}\n')