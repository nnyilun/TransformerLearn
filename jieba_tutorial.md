### 基础使用

`jieba` 支持三种分词模式：精确模式、全模式和搜索引擎模式。

#### 精确模式

精确模式试图将句子最精确地切开，适合文本分析。

```python
import jieba

text = "我来到北京清华大学"
seg_list = jieba.cut(text, cut_all=False)
print("精确模式: " + "/ ".join(seg_list))
```

#### 全模式

全模式会将句子中所有可能的词语都扫描出来，速度非常快，但不适合文本分析。

```python
seg_list = jieba.cut(text, cut_all=True)
print("全模式: " + "/ ".join(seg_list))
```

#### 搜索引擎模式

搜索引擎模式，在精确模式的基础上，对长词再次切分，适合用于搜索引擎分词。

```python
seg_list = jieba.cut_for_search(text)
print("搜索引擎模式: " + "/ ".join(seg_list))
```

### 添加自定义词典

为了提高分词的准确性，你可以添加自定义词典来帮助 `jieba` 更好地分词。

```python
jieba.load_userdict("userdict.txt")
```

其中，`userdict.txt` 是一个自定义词典的文件，每行包含一个词和该词的权重，用空格隔开。

### 调整词典

`jieba` 允许动态调整词典，如添加或删除某个词。

```python
jieba.add_word('清华大学')
jieba.del_word('自定义词')
```