import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from utils import accuracy
from module import SentimentModel
from data import CommentSet, padding_sentences, make_vocab, apply_tokenizer, read_comment_csv, concat_dataframe

cpu_threads = torch.get_num_threads()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'cpu thread num: {cpu_threads}')
print(f'device is {device}, cuda_device_0:{torch.cuda.get_device_name(0)}')

df1 = read_comment_csv("../dataset/DMSC_2M.csv", 10000)
df2 = read_comment_csv("../dataset/DMSC_10M.csv", 1)
combined_df = concat_dataframe(df1, df2)

tokenized = apply_tokenizer(combined_df["Comment"], use_stopwords=True)
vocab = make_vocab(tokenized, min_freq=4)
print(len(vocab))

df_length = tokenized.apply(len)
filtered_df = tokenized[df_length < 64]
num_sentences_less_than = filtered_df.shape[0]
print(num_sentences_less_than, f'{num_sentences_less_than / len(tokenized) * 100}%')

padded_sequences = padding_sentences(vocab, tokenized, target_len=64)

comment_set = CommentSet(padded_sequences, combined_df["Star"])
train_data = DataLoader(comment_set, batch_size=32, shuffle=True, num_workers=cpu_threads, pin_memory=True)

num_epochs = 10
num_classes = 3
learning_rate = 0.01
vocab_len = len(vocab)
d_model = 1024
num_heads = 16
n_layers = 16
d_ff = 4096

model = SentimentModel(vocab_len, d_model, num_heads, d_ff, n_layers, num_classes, dropout=0.4).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

swa_model = AveragedModel(model)
swa_start = 80
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
swa_scheduler = SWALR(optimizer, swa_lr=0.0001)

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    results = []
    for X, y in tqdm(train_data):
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

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_data)}, Accuracy: {sum(results) / len(results)}")

torch.optim.swa_utils.update_bn(train_data, swa_model)
swa_model.eval()
results = []
with torch.no_grad():
    for X, y in train_data:
        X, y = X.to(device), y.to(device)
        
        with autocast():
            output = model(X)
        
        results.append(accuracy(output, y))
    
        if len(results) > 100:
            break

print(f'evaluate acc: {sum(results) / len(results)}')

torch.save(model.state_dict(), '../model/model_statedict1214_01.pt')