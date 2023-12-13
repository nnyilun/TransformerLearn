# Transformer学习

目前仅编写了Encoder，进行电影评论数据的情感分析。

## 数据集

在Kaggle上找到的豆瓣电影评论数据集

* 10M大小

[Dou ban Movie short comments (10377Movies)](https://www.kaggle.com/datasets/liujt14/dou-ban-movie-short-comments-10377movies)

* 2M大小

[Douban Movie Short Comments Dataset](https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments?rvi=1)

### 停用词表

[goto456/stopwords: 中文常用停用词表（哈工大停用词表、百度停用词表等）](https://github.com/goto456/stopwords)

（但是停用词表中存在一些在情感分析中不应该删除的词语）

| 词表名 | 词表文件 |
| - | - |
| 中文停用词表                   | cn\_stopwords.txt    |
| 哈工大停用词表                 | hit\_stopwords.txt   |
| 百度停用词表                   | baidu\_stopwords.txt |
| 四川大学机器智能实验室停用词库 | scu\_stopwords.txt   |

### 数据预处理

* 使用jieba进行分词，可以选择是否使用停用词表。

* 构建Vocab，按照一定词频限制

* 将所有输入padding到相同长度

* 标签规定：1-2分为负面，3为中性，4-5为正面

## 模型

使用pytorch，参照`Attention is all your need`论文搭建。未使用预训练模型。

[[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 训练

使用RTX4060Ti进行小规模数据集测试，使用RTX4090进行完整训练。