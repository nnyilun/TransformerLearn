from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR, AveragedModel
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
from data import read_comment_csv, concat_dataframe

pandarallel.initialize(progress_bar=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32
max_len = 64
num_epochs = 200
learning_rate = 1e-5
final_learning_rate = 1e-7
num_labels = 3
train_period = 20
swa_start = 180
pretrain_model_path = "../model/bert-base-chinese"

config = BertConfig.from_pretrained(
    pretrain_model_path,
    num_labels=num_labels,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)


class CommentSet(Dataset):
    def __init__(self, texts:pd.Series, labels:pd.Series, tokenizer:BertTokenizer, max_len:int=128) -> None:
        super().__init__()
        assert len(texts) == len(labels), "inputs and labels do not have the same lenght!"
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, index:int) -> Any:
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'review_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

def create_data_loader(
        texts:pd.Series, 
        labels:pd.Series, 
        tokenizer:BertTokenizer, 
        max_len:int, 
        batch_size:int, 
        shuffle:bool=False,
        num_workers:int=12, 
        pin_memory:bool=True
    ) -> DataLoader:
    ds = CommentSet(
        texts=texts.to_numpy(),
        labels=labels.to_numpy(dtype=np.int64),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    

def save_checkpoint(model, optimizer, scheduler, swa_model, swa_scheduler, epoch, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'swa_state_dict': swa_model.state_dict() if swa_model is not None else None,
        'swa_scheduler': swa_scheduler.state_dict() if swa_scheduler is not None else None,
    }
    torch.save(checkpoint, filename)


def save_model(model, filename="model.pth.tar"):
    torch.save(model.state_dict(), filename)


def evaluate(model:BertForSequenceClassification, dataloader:DataLoader, device:str="cuda") -> float:
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data in dataloader:
            input_ids, labels, attention_mask = data["input_ids"].to(device), data["labels"].to(device), data["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')
    model.train()
    return accuracy


def train(
        model:BertForSequenceClassification, 
        optimizer:torch.optim.Optimizer, 
        dataloader:DataLoader, 
        num_epochs:int, 
        device:str,
        valid_loader:DataLoader=None,
        scheduler:CosineAnnealingLR=None,
        swa_model:AveragedModel=None,
        swa_scheduler:SWALR=None,
        swa_start:int=5
    ) -> None:
    model.train()
    max_valid_acc = 0.0
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, data in progress_bar:
            input_ids, labels, attention_mask = data["input_ids"].to(device), data["labels"].to(device), data["attention_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)

            progress_bar.set_description(f"Epoch {epoch+1}")
            progress_bar.set_postfix(loss=total_loss/(i+1), accuracy=100.*correct/total, Learning_rate=optimizer.param_groups[0]['lr'])

            if epoch >= swa_start and swa_model is not None:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler is not None:
                scheduler.step()

        if valid_loader is not None:
            valid_acc = evaluate(model, valid_loader, device)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                save_checkpoint(model, optimizer, scheduler, swa_model, swa_scheduler, epoch, filename=f"../model/bert_result/checkpoint_epoch_{epoch+1}.pth.tar")

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy}%")
    
    print(f"Done! max valid accuracy: {max_valid_acc}")


def get_sentiment(x:int) -> int:
    if 1 <= x <= 2:
        return 0
    elif x == 3:
        return 1
    elif 4 <= x <= 5:
        return 2
    else:
        raise ValueError("Unknown value!")


if __name__ == "__main__":
    df1 = read_comment_csv("../dataset/DMSC_2M.csv", nrows=10000)
    df2 = read_comment_csv("../dataset/DMSC_10M.csv", nrows=1)
    combined_df = concat_dataframe(df1, df2)

    combined_df["Sentiment"] = combined_df["Star"].parallel_apply(get_sentiment)

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(combined_df["Comment"], combined_df["Sentiment"], test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    train_loader = create_data_loader(train_texts, train_labels, tokenizer, max_len, batch_size)
    valid_loader = create_data_loader(valid_texts, valid_labels, tokenizer, max_len, batch_size)

    model = BertForSequenceClassification(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - swa_start)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=final_learning_rate)

    train(model, optimizer, train_loader, num_epochs, device, valid_loader, scheduler, swa_model, swa_scheduler, swa_start)

    save_model(model, filename="../model/bert_result/20231220_model.pth.tar")