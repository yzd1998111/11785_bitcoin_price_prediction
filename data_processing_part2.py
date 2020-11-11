import pandas as pd
import numpy as np
from textblob import TextBlob

# price
price_data = pd.read_csv("bitstampUSD.csv")
price_data = price_data.dropna()
price_data = price_data.sort_values(by='Timestamp')
price_data['Date']=pd.to_datetime(price_data['Timestamp'],unit='s')
price_data['Date']=price_data['Date'].dt.date
date_set=set(price_data['Date'].values)
date_set = sorted(date_set)
output_dict={'Timestamp':[],
             'Close':[],
             'Change':[],
             'Volume':[]
            }

for i in range(len(date_set)):
    temp_data=price_data[price_data['Date']==date_set[i]]
    if i == 0:
        prev_close=temp_data['Close'].values[-1]
    if i != 0:
        cur_close=temp_data['Close'].values[-1]
        output_dict['Timestamp'].append(date_set[i])
        output_dict['Close'].append(cur_close)
        output_dict['Change'].append(1 if (cur_close-prev_close)>0 else 0)
        output_dict['Volume'].append(temp_data['Volume_(Currency)'].sum())
        prev_close=cur_close

output_data=pd.DataFrame(output_dict)
output_data.to_csv('price_data_w_V.csv',index=False)



#calculate sentiment
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

#investing.com
investing = pd.read_csv("investing_final.csv")
investing['polarity'] = investing['Comment'].apply(pol)
investing['subjectivity'] = investing['Comment'].apply(sub)
investing_date = list(set(investing['Timestamp'].values))
investing_date.sort()
num_com = []
avg_sent = []
avg_subj = []

for d in investing_date:
    try:
        temp_data = investing[investing['Timestamp'] == d]
        num_com.append(len(temp_data))
        avg_sent.append(temp_data['polarity'].mean())
        avg_subj.append(temp_data['subjectivity'].mean())
        
    except:
        print('except:', d)
        
investing_summary_dict = {
    'date': investing_date,
    'num_com': num_com,
    'avg_sent': avg_sent,
    'avg_subj': avg_subj
}

investing_summary = pd.DataFrame(investing_summary_dict)
investing_summary.to_csv("investing_summary.csv")

#reddit

reddit_eng_1 = pd.read_csv("reddit_eng_20120101to20171231_train.csv")
reddit_eng_2 = pd.read_csv("reddit_eng_20180101to20190430_val.csv")
reddit_eng_3 = pd.read_csv("reddit_eng_20190501to20191123_test.csv")
reddit = pd.concat([reddit_eng_1, reddit_eng_2, reddit_eng_3], ignore_index = True)

reddit['polarity'] = reddit['Comment'].apply(pol)
reddit['subjectivity'] = reddit['Comment'].apply(sub)

reddit_date = list(set(reddit['Timestamp'].values))
reddit_date.sort()

num_com = []
avg_sent = []
avg_subj = []

for d in reddit_date:
    try:
        temp_data = reddit[reddit['Timestamp'] == d]
        num_com.append(len(temp_data))
        avg_sent.append(temp_data['polarity'].mean())
        avg_subj.append(temp_data['subjectivity'].mean())
    except:
        print('except:', d)

reddit_summary_dict = {
    'date': reddit_date,
    'num_com': num_com,
    'avg_sent': avg_sent,
    'avg_subj': avg_subj
}

reddit_summary = pd.DataFrame(reddit_summary_dict)

reddit_summary.to_csv("reddit_summary.csv")
