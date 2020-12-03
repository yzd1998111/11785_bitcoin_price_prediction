import pandas as pd
from langdetect import detect
import re
import matplotlib.pyplot as plt
import nltk
import torch
from models import InferSent
from nltk.corpus import stopwords

nltk.download('stopwords')

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
W2V_PATH = 'fastText/crawl-300d-2M.vec'

infersent_model = InferSent(params_model)
infersent_model.load_state_dict(torch.load(MODEL_PATH))
infersent_model.set_w2v_path(W2V_PATH)
infersent_model.build_vocab_k_words(K=100000)


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def clean_text(x):
    x = re.sub(r'[?|$|.|!]',r'',x) #for remove particular special char
    x = re.sub(r'[^a-zA-Z0-9 ]',r'',x) #for remove all characters
    x =''.join(c if c not in map(str,range(0,10)) else '' for c in x) #for remove numbers
    x = re.sub('  ',' ',x) #for remove extra spaces
    return x

def stopword_removal(x):
    return " ".join([word for word in x.split() if word and word not in stopwords.words('english')])

def lower(x):
    return x.lower()

total_len = total_cnt = 0
total_coverage = []
def filter_fn(row):
    global total_len, total_cnt
    try:
        return detect(row["Comment"]) == "en" and len(row["Comment"]) >= 100 and len(row["Comment"]) <= 500
    except:
        return False


def label_coverage(row):
    sent = row["Comment"].split()
    coverage = sum([1 for word in sent if word in infersent_model.word_vec])
    return (coverage/len(sent)*100) if sent else 0

print("loading data")
df = pd.read_csv("comment/investing_raw.csv", header=0)
df = df.dropna()
print("lowering")
df['Comment'] = df['Comment'].apply(lower)
df['Comment'] = df['Comment'].apply(deEmojify)
df['Comment'] = df['Comment'].apply(replace_contractions)
#df['Comment'] = df['Comment'].apply(stopword_removal)
df['Comment'] = df['Comment'].apply(clean_text)
df['Coverage'] = df.apply (lambda row: label_coverage(row), axis=1)

print("filtering")
df = df[df.apply(filter_fn, axis=1)]
print(df.head(5))

#plt.hist(total_coverage, color = 'blue', edgecolor = 'black',bins=50)
#plt.savefig("result.png")

df = df.groupby('Timestamp').apply(lambda x: x.sort_values('Coverage', ascending=False).head(30))
df.drop_duplicates(subset=['Comment'], inplace=True)
df.to_csv("comment/investing.csv", index=None)
