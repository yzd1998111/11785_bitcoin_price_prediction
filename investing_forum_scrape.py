from datetime import timedelta, date
import time
import collections
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import re
import requests

PAGES = 3
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)



start_date = date(2018, 1, 1)
# end_date = date(2019, 11, 24)
end_date = date(2018, 1, 2)
years = set()
bad_page = 0

regex = datetime.datetime.strptime

result = []
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36',
    'Content-Type': 'text/html',
}


for i in range(0, 4281, 1):
    if i % 100 == 0:
        print(f"working on page: {i}")
        print(bad_page)
        cur_year_df = pd.DataFrame(result, columns =['Timestamp', 'Comment'])
        cur_year_df.drop_duplicates(subset=['Comment'], inplace=True)
        cur_year_df.to_csv(f"talk.csv",index=False)

    url = f"https://www.investing.com/crypto/bitcoin/chat/{i}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    comments = []
    time_stamps = []
    for tmp in soup.findAll("div", {'class': ["js-text-wrapper commentText", "js-text-wrapper commentText withImage"]}):
        for result_table in tmp.findAll('span', {"class": "js-text"}):
            if result_table.getText() != "{commentContent}":
                comments.append(result_table.getText().lower().strip())
          
    for result_table in soup.findAll('span', {'class': "js-date"}):
        date = result_table["comment-date"]
        mat1= re.match('\d{4}[-/]\d{2}[-/]\d{2}', date)
        if mat1:
            time_stamps.append(date[mat1.span()[0]:mat1.span()[1]])

    if len(comments) != len(time_stamps):
        bad_page += 1
        comments = comments[:len(time_stamps)]
        time_stamps = time_stamps[:len(comments)]

    assert(len(comments)==len(time_stamps))
    # print(comments)
    result.extend([t for t in zip(time_stamps,comments)])
    time.sleep(2)




