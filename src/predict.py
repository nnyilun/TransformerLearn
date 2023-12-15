import torch
from module import *
from data import *

device = 'cpu'
print(f'device is {device}')

vocab = load_vocab(path="../model/2M_vocab_freq16_nouseStopwords.pkl") 
print(len(vocab)) # vocab_len(47088)
state_dict = torch.load("../model/model_statedict1212_02.pt")

num_classes = 3
vocab_len = len(vocab)
src_pad_idx = vocab['<pad>']
max_len = 64
d_model = 512
num_heads = 8
d_ff = 2048
n_layers = 6

model = TransformerForSentimentAnalysis(vocab_len, max_len, d_model, num_heads, d_ff, n_layers, src_pad_idx, num_classes).to(device=device)
model.load_state_dict(state_dict)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable_params: {trainable_params}")
print(f"non_trainable_params: {non_trainable_params}")

model.eval()

while True:
    usr_input = input("Enter a comment(enter q to quit):")

    if usr_input == 'q':
        break

    usr_input = tokenize_str(usr_input, use_stopwords=False)
    print(f'token result:{usr_input}')
    usr_input = padding_one_sentence(vocab, usr_input)
    print(f'padding input: {usr_input}')

    predict = model(usr_input)
    print(predict, torch.argmax(predict))
