import csv
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import langdetect as ld
from torch import nn
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset, DataLoader
from models import InferSent
import collections
import string
import nltk
nltk.download('punkt')

sentences = ["helloword", "dog", "dog2"]
sentences2 = ["cat", "kitten"]
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
W2V_PATH = 'fastText/crawl-300d-2M.vec'



torch.cuda.empty_cache()
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
EPOCHS = 10
HIDDEN_DIM = 512
BATCH_SIZE = 64

IN_FEATURES = 13
OUT_FEATURES = 2

class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_size * 2, out_features)
    

    def forward(self, X, lengths):
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False, batch_first=True)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(lstm_out)
        out = self.output(out).log_softmax(2)
        return out, out_lens


class BitcoinDataset(Dataset):
    def __init__(self, comment_file = "", price_file = "", transform=None):
        self.comment_file = comment_file
        self.price_file = price_file

        # self.indices_date_map = {} 
        self.comment_dict = collections.defaultdict(list) 
        self.X = collections.defaultdict(list) 
        self.X_lens = []
        self.Y = []

        self.infersent = InferSent(params_model)
        self.infersent.load_state_dict(torch.load(MODEL_PATH))
        self.infersent.set_w2v_path(W2V_PATH)
        self.infersent.build_vocab_k_words(K=500000)

        self.comment = pd.read_csv(comment_file, encoding='utf-8') if comment_file else None
        self.price = pd.read_csv(price_file, encoding='utf-8') if price_file else None

        self.size = self.price.shape[0]
        self.max_len = 0
        for idx, row in self.price.iterrows():
            self.Y.append(row["Change"])

        for idx, row in self.comment.iterrows():
            date = row["Timestamp"]
            self.comment_dict[date].append(row["Comment"])

        for k, comments in self.comment_dict.items():
            self.X[k] = self.infersent.encode(comments, tokenize=True)
            self.X_lens.append(len(comments))
            self.max_len = max(self.max_len, len(comments))

        print(self.max_len)
        self.X = pad_sequence(self.X.values(), batch_first=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # date = self.indices_date_map[idx]
        return (self.X[idx], self.X_lens[idx], self.Y[idx])
        

# choose the training and test datasets
train_data = BitcoinDataset('reddit_dev.csv', 'price_dev.csv')
val_data = BitcoinDataset('reddit_dev.csv', 'price_dev.csv')
test_data = BitcoinDataset('reddit_dev.csv', 'price_dev.csv')

# prepare data loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False)


model = Model(IN_FEATURES, OUT_FEATURES, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)


def train():
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0.0
        print(f"starting epoch: {epoch}")
        cnt = 0
        for X, X_lens, Y in train_loader:
            X, X_lens, Y, Y_lens = X.cuda(), X_lens.cuda(), Y.cuda()
            optimizer.zero_grad()
            out, out_lens = model(X, X_lens)
            loss = criterion(out.transpose(0, 1), Y, out_lens)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()     
            if cnt % 50 == 0:
                print(f"progress: {cnt} / {len(train_loader)}")
            cnt += 1
        print('Epoch', epoch + 1, 'Loss', loss.item())
        validate()
        model.train()
    torch.save(model.state_dict(), "./model.pth")
    test()

# def validate():
#     total_dist = 0
#     model.eval()
#     decoder = CTCBeamDecoder(['$'] * (N_PHONEMES + 1), beam_width=10, log_probs_input=True)

#     with torch.no_grad():
#         cnt = 0
#         for X, X_lens, Y, Y_lens in val_loader:
#             X, X_lens, Y, Y_lens = X.cuda(), X_lens.cuda(), Y.cuda(), Y_lens.cuda()
#             out, out_lens = model(X, X_lens)
#             # print("hehe", out.shape, out_lens, out_lens.shape)
#             val_Y, _, _, val_Y_lens = decoder.decode(out, out_lens)

#             for i in range(BATCH_SIZE):
#                 cnt += 1
#                 best_seq = val_Y[i, 0, :val_Y_lens[i,0]]
#                 best_pron = ''.join(PHONEME_MAP[j] for j in best_seq)
#                 actual_seq = Y[i, :Y_lens[i]]
#                 actual_pron = ''.join(PHONEME_MAP[j] for j in actual_seq)
#                 total_dist += lev(best_pron, actual_pron)

               
#     lv_dist = total_dist / cnt
#     print(lv_dist)
  

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