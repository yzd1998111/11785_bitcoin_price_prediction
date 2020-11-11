import csv
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset, DataLoader
from models import InferSent
from matplotlib import pyplot as plt
from tqdm import tqdm
import collections
import string
import time
import nltk
import sys
import time
from torch.nn import Parameter
from functools import wraps
from datetime import datetime, timedelta
nltk.download('punkt')

V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
W2V_PATH = 'fastText/crawl-300d-2M.vec'

infersent_model = InferSent(params_model)
infersent_model.load_state_dict(torch.load(MODEL_PATH))
infersent_model.set_w2v_path(W2V_PATH)

start = time.time()
comments_df = pd.read_csv("comment/investing.csv", encoding='utf-8')
comments = comments_df["Comment"].values
infersent_model.build_vocab(comments, tokenize=True)
print(f"constructing comment embedding for {len(comments)} comments")
comment_embedding = infersent_model.encode(comments, tokenize=True)
comment_embedding_dict = {comments[i]: comment_embedding[i] for i in range(len(comments))}
print("embedding construction completed", time.time()-start)
torch.cuda.empty_cache()
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
EPOCHS = 20
HIDDEN_DIM = 256
BATCH_SIZE = 2
CONTEXT_SIZE = 4

IN_FEATURES = 4096
OUT_FEATURES = 2

BLCK1_FILTER = 128
BLCK2_FILTER = 256

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv_tmp = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        self.bn_tmp = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = self.bn_tmp(self.conv_tmp(x)) if self.in_channels != self.out_channels else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + residual)
        return x

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training))
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)



class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.num_layers = 3
        
        self.res1 = ResBlock(in_features, BLCK1_FILTER)
        self.bn1 = nn.BatchNorm1d(BLCK1_FILTER)
        self.res2 = ResBlock(BLCK1_FILTER, BLCK1_FILTER)
        self.bn2 = nn.BatchNorm1d(BLCK1_FILTER)
        self.res3 = ResBlock(BLCK1_FILTER, BLCK1_FILTER)
        self.bn3 = nn.BatchNorm1d(BLCK1_FILTER)
        
        self.res4 = ResBlock(BLCK1_FILTER, BLCK2_FILTER)
        self.bn4 = nn.BatchNorm1d(BLCK2_FILTER)
        self.res5 = ResBlock(BLCK2_FILTER, BLCK2_FILTER)
        self.bn5 = nn.BatchNorm1d(BLCK2_FILTER)
        self.res6 = ResBlock(BLCK2_FILTER, BLCK2_FILTER)
        self.bn6 = nn.BatchNorm1d(BLCK2_FILTER)

        self.lstm = WeightDrop(torch.nn.LSTM(BLCK2_FILTER, self.hidden_size, self.num_layers, batch_first=True), [f'weight_hh_l{i}' for i in range(self.num_layers)], dropout=0.65)


        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, out_features)
    

    def forward(self, X, lengths):
        batch_size, seq_len, embedding_dim = X.shape
        X = X.transpose(1,2)
        X = self.bn1(self.res1(X))
        X = self.bn2(self.res2(X))
        X = self.bn3(self.res3(X))

        X = self.bn4(self.res4(X))
        X = self.bn5(self.res5(X))
        X = self.bn6(self.res6(X))
        X = X.transpose(1,2)

        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False, batch_first=True)
        packed_out = self.lstm(packed_X)[0]

        out, out_lens = pad_packed_sequence(packed_out, batch_first=True)

        seq_indicies = [l - 1 for l in out_lens]
        batch_indicies = [i for i in range(batch_size)]
        out = out[batch_indicies,seq_indicies,:]

        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out, out_lens


class BitcoinDataset(Dataset):
    def __init__(self, comment_file = "", price_file = "", transform=None):
        self.comment_file = comment_file
        self.price_file = price_file

        self.comment_dict = collections.defaultdict(list) 
        self.X = []
        self.X_lens = []
        self.Y_tmp = {}
        self.Y = []
        

        self.comment = pd.read_csv(comment_file, encoding='utf-8') if comment_file else None
        self.comment_embedding = []
        self.comments = []
        self.price = pd.read_csv(price_file, encoding='utf-8') if price_file else None

        self.max_len = 0
        self.min_len = sys.maxsize
        self.size = 0

        start_date = datetime.strptime("2017-09-15", "%Y-%m-%d")
        first_date, last_date = "", ""

        for idx, row in self.price.iterrows():
            cur_date = datetime.strptime(row["Timestamp"], "%Y-%m-%d")
            if cur_date >= start_date:
                if len(first_date) == 0:
                    first_date = (cur_date - timedelta(days=CONTEXT_SIZE)).strftime("%Y-%m-%d")
                last_date = row["Timestamp"]
                self.Y_tmp[row["Timestamp"]] = int(row["Change"])

        self.comment = self.comment[(self.comment["Timestamp"]>= first_date) & (self.comment["Timestamp"]<=last_date)]
        
        for idx, row in self.comment.iterrows():
            cur_date = datetime.strptime(row["Timestamp"], "%Y-%m-%d") 
            for i in range(CONTEXT_SIZE):
                next_date = cur_date + timedelta(days=i)
                self.comment_dict[next_date.strftime("%Y-%m-%d")].append(row["Comment"])
            self.comments.append(row["Comment"])

        for k in sorted(self.Y_tmp.keys()):
            self.size += 1

            context_comments = self.comment_dict[k]
            context_comments_embeddings = []
            for comment in context_comments:
                context_comments_embeddings.append(comment_embedding_dict[comment][:])

            self.X.append(torch.Tensor(context_comments_embeddings))
            # self.X.append(context_comments)
            self.X_lens.append(len(context_comments))
            self.max_len = max(self.max_len, len(context_comments))
            self.min_len = min(self.min_len, len(context_comments))


            self.Y.append(torch.tensor(self.Y_tmp[k]))
 
        self.X = pad_sequence(self.X, batch_first=True)
        del self.comments, self.Y_tmp, self.comment_dict, self.price, self.comment
        print(self.max_len, self.min_len)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.X[idx], self.X_lens[idx], self.Y[idx])
        

# choose the training and test datasets
train_data = BitcoinDataset('comment/investing.csv', 'price/price_train.csv')
val_data = BitcoinDataset('comment/investing.csv', 'price/price_val.csv')
# test_data = BitcoinDataset('talk.csv', 'price_dev.csv')

# prepare data loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False, drop_last=True)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False)

model = Model(IN_FEATURES, OUT_FEATURES, HIDDEN_DIM).cuda()
print(model.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)


def train():
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_correct = 0
        print(f"starting epoch: {epoch + 1}, batches: {len(train_loader)}")
        cnt = 0
        for X, X_lens, Y in tqdm(train_loader):
            X, X_lens, Y = X.cuda(), X_lens.cpu(), Y.cuda()
            optimizer.zero_grad()
            out, out_lens = model(X, X_lens)

            loss = criterion(out, Y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pred = torch.argmax(out, dim=1)
            train_correct += torch.sum(torch.eq(pred, Y))     
          
            cnt += 1

        print('Epoch', epoch + 1, 'Train Loss', train_loss, 'Correct', train_correct / (1.0 * len(train_loader.dataset)))
        val_loss, val_acc = validate()
        train_acc = train_correct / (1.0 * len(train_loader.dataset))
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item())
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())
        model.train()
    torch.save(model.state_dict(), "./model.pth")
    # test()
    plt.figure(0)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training losses')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.legend()
    plt.savefig('lstm_loss.png')
    
    plt.figure(1) 
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training accuracy')
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))
    plt.legend()
    plt.savefig('lstm_acc.png')


def validate():
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        for X, X_lens, Y in val_loader:
            X, X_lens, Y = X.cuda(), X_lens.cpu(), Y.cuda()
            optimizer.zero_grad()
            out, out_lens = model(X, X_lens)

            loss = criterion(out, Y)
            val_loss += loss.item()

            pred = torch.argmax(out, dim=1)

            val_correct += torch.sum(torch.eq(pred, Y))     
        val_acc = val_correct / (1.0 * len(val_loader.dataset))
        print('Val Loss', val_loss, 'Correct', val_acc)
        return val_loss, val_acc

# def test():
#     model.eval() # prep model for *evaluation*
#     decoder = CTCBeamDecoder(['$'] * (N_PHONEMES + 1), beam_width=10, log_probs_input=True)
#     with torch.no_grad():
#         with open('submission.csv', mode='w') as submission:
#             writer = csv.writer(submission, delimiter=',')
#             row = 0
#             writer.writerow(['id', 'label'])
#             for X, X_lens in test_loader:
#                 X, X_lens = X.cuda(), X_lens.cuda()
#                 out, out_lens = model(X, X_lens)
#                 test_Y, _, _, test_Y_lens = decoder.decode(out, out_lens)
#                 for i in range(BATCH_SIZE):
#                     best_seq = val_Y[i, 0, :val_Y_lens[i,0]]
#                     best_pron = ''.join(PHONEME_MAP[j] for j in best_seq)
#                     writer.writerow([row, best_pron])
#                     row += 1


if __name__ == "__main__":
    train()