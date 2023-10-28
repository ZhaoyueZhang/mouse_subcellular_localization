import torch
import gc

import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
import sys
import random
import os

from Bio import SeqIO
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from skmultilearn.model_selection import IterativeStratification
from torch.optim.lr_scheduler import ReduceLROnPlateau

#import wandb
import warnings
warnings.filterwarnings("ignore")

gc.collect()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

#wandb.init(project="3seed1")
set_seed(0)#wandb.init(project="seed")


# hyperparameters
MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 40#wandb.config.epoch
LEARNING_RATE = 5e-5
NUMS_LABELS = 3
OUTPUT_SIZE = 3
LABEL_LENGTH = 3
seq_len = 2048#wandb.config.pads
dropout_n=0.2


def create_dataframe(string_list):
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each string in the list
    for string in string_list:
        # Create a dictionary to hold the letters of the string
        string_dict = {}

        # Iterate over each letter in the string
        for i, letter in enumerate(string):
            # Create column name (e.g., 'Letter 1', 'Letter 2', etc.)
            col_name = f'label{i+1}'

            # Add the letter to the dictionary
            string_dict[col_name] = letter

        # Append the dictionary as a row to the DataFrame
        tdf = pd.DataFrame(string_dict,index=[0])
        df = pd.concat([df, tdf],ignore_index=True)
    return df


def truncate_sequence(sequence, max_length):
    if len(sequence) <= max_length:
        return sequence
    else:
        return sequence[-max_length:]

def process_dataset(dataset, max_length=2048):
    processed_dataset = []
    for sequence in dataset:
        processed_sequence = truncate_sequence(sequence, max_length)
        processed_dataset.append(processed_sequence)
    return np.array(processed_dataset)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

def loss_fn(outputs, targets):

    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

val_targets=[]
val_outputs=[]

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df_x, df_y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df_x = df_x
        self.df_y = df_y
        self.max_len = max_len

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, index):
        df_x = str(self.df_x[index])
        df_x = " ".join(df_x.split())

        inputs = self.tokenizer.encode_plus(
            df_x,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'features': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.FloatTensor(self.df_y[index])
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(dropout_n)
        self.linear = torch.nn.Linear(768, LABEL_LENGTH)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        if input_ids.shape[1] > MAX_LEN:
            num_sub_sequences = (input_ids.shape[1] - 1) // MAX_LEN + 1
            outputs = []
            for i in range(num_sub_sequences):
                start_idx = i * MAX_LEN
                end_idx = min((i + 1) * MAX_LEN, input_ids.shape[1])
                sub_input_ids = input_ids[:, start_idx:end_idx]
                sub_attn_mask = attn_mask[:, start_idx:end_idx]
                sub_token_type_ids = token_type_ids[:, start_idx:end_idx]

                output = self.bert_model(
                    input_ids=sub_input_ids, 
                    attention_mask=sub_attn_mask, 
                    token_type_ids=sub_token_type_ids
                )
                output_dropout = self.dropout(output.pooler_output)
                outputs.append(output_dropout)
            #print(outputs)
            output_final = torch.cat(outputs, dim=1)
            output = self.linear(output_final)
        
        else:
            output = self.bert_model(
                input_ids=input_ids, 
                attention_mask=attn_mask, 
                token_type_ids=token_type_ids
            )
            output_dropout = self.dropout(output.pooler_output)
            output = self.linear(output_dropout)

        return output


filename = "mus_seq_label3.fasta"
sequences = SeqIO.parse(filename, "fasta")

X, y = [], []
for record in sequences:
    output = ' '.join(record.seq)
    X.append(output)
    y.append(record.id[:LABEL_LENGTH])

df = create_dataframe(y)

col_name = [f'label{i+1}' for i in range(LABEL_LENGTH)]
for col_n in col_name:
    df[col_n] = df[col_n].astype(str).astype(int)

df.insert(loc=0, column='sequence', value=X)
pd.set_option('display.max_rows', df.shape[0]+1)

LABEL_COLUMNS = ['label1', 'label2', 'label3']
df[LABEL_COLUMNS].sum()

X, y = df['sequence'], df[LABEL_COLUMNS]
X, y = X.to_numpy(), y.to_numpy()
X = process_dataset(X, max_length=seq_len)
foldperf={}
fold = 0

k_fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train, test in k_fold.split(X, y):
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    train_dataset = CustomDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = CustomDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
        )

    model = BERTClass()
    model.to(device)
   
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5,min_lr=0.00001)

    history = {'test_loss': [], 'test_acc':[], 'test_micro': [], 'test_ap': []}

    for epoch in range(1, EPOCHS+1):
        train_loss = 0
        test_loss = 0
        train_correct = 0
        test_correct = 0

        model.train()
        for batch_idx, data in enumerate(train_loader):
            ids = data['features'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += ((1 / (batch_idx + 1)) * loss.item())

            # Calculate training accuracy
            predicted_labels = torch.round(torch.sigmoid(outputs))
            train_correct += torch.sum(predicted_labels == labels).item()
        
        model.eval()
        with torch.no_grad():
            test_targets = []
            test_outputs = []
            for batch_idx, data in enumerate(test_loader):
                features = data['features'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                labels = data['labels'].to(device, dtype=torch.float)

                outputs = model(features, mask, token_type_ids)

                loss = loss_fn(outputs, labels)
                test_loss = test_loss + ((1 / (batch_idx + 1)) * loss.item())

                # Calculate validation accuracy
                predicted_labels = torch.round(torch.sigmoid(outputs))
                test_correct += torch.sum(predicted_labels == labels).item()

                test_targets.extend(labels.cpu().detach().numpy().tolist())
                test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                
            # Calculate average losses and accuracies
            test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct / (len(test_loader.dataset) * labels.shape[1])

            # Calculate ROC scores
            test_targets = np.array(test_targets)
            test_outputs = np.array(test_outputs)
            # try:
                # test_roc_macro = roc_auc_score(test_targets, test_outputs, average='macro')
            # except  ValueError:
                # print(test_targets)
                # break
            test_roc_micro = roc_auc_score(test_targets, test_outputs, average='micro')
            test_ap = average_precision_score(test_targets, test_outputs)

            # Print training/validation statistics
            print("Fold {} Epoch:{}/{} Loss:{:.4f} Accuracy:{:.4f} ROC Micro: {:.4f} ap: {:.4f}"
                    .format(fold+1, epoch, EPOCHS, test_loss, test_accuracy, test_roc_micro, test_ap))#test_roc_macro, 

            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_accuracy)
            #history['test_macro'].append(test_roc_macro)
            history['test_micro'].append(test_roc_micro)
            history['test_ap'].append(test_ap)
            
    foldperf['fold{}'.format(fold+1)] = history 
    fold += 1
  
roc_micro_mean = np.mean([history['test_micro'][-1] for history in foldperf.values()])

#test_loss, test_accuracy, test_macro, test_micro, test_ap = [], [], [], [], []
test_loss, test_accuracy, test_micro, test_ap = [], [], [], []
k=5
for f in range(1,k+1):

     test_loss.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))
     test_accuracy.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

     #test_macro.append(np.mean(foldperf['fold{}'.format(f)]['test_macro']))
     test_micro.append(np.mean(foldperf['fold{}'.format(f)]['test_micro']))
     test_ap.append(np.mean(foldperf['fold{}'.format(f)]['test_ap']))

print('Performance of {} fold cross validation'.format(k))
print("Average Loss: {:.4f} \t Accuracy: {:.4f} \t Micro: {:.4f} \t ap: {:.4f}".format(np.mean(test_loss),np.mean(test_accuracy),np.mean(test_micro),np.mean(test_ap)))
wandb.log({"Loss": np.mean(test_loss)})
wandb.log({"Acc": np.mean(test_accuracy)})
wandb.log({"ROC_micro": np.mean(test_micro)})
wandb.log({"AP": np.mean(test_ap)})
g = open('mus_seed_label3_1028.txt','a')
g.write("{:d}\t{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(seq_len,EPOCHS,np.mean(test_loss),np.mean(test_accuracy),np.mean(test_micro),np.mean(test_ap)))
#MODEL_SAVE_PATH = "bert_model.pt"
#torch.save(model.state_dict(), MODEL_SAVE_PATH)

#wandb sweep lnc_wandb.yaml
#copy yellow code run