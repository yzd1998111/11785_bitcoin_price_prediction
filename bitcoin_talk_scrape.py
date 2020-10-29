from GoogleNews import GoogleNews
from newspaper import Article
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


regex = datetime.datetime.strptime
# url = "https://bitcointalk.org/index.php?board=77.34280"
result = []
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36',
    'Content-Type': 'text/html',
}


for i in range(0, 34281, 40):
    if i % 400 == 0:
        print(f"working on page: {i/40}")
    url = f"https://bitcointalk.org/index.php?board=77.{i}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')


    for result_table in soup.findAll('td', {'class': 'windowbg'}):
        if result_table.find('span'):
            title = result_table.find('span').getText()
            # print(title)
            title = title.strip()
            # print("hehe", title)
            mat1= re.match('\d{4}[-/]\d{2}[-/]\d{2}', title)
            mat2 = re.match('\[\d{4}[-/]\d{2}[-/]\d{2}\]', title)
            if mat1:
                result.append([mat1[0],title[mat1.span()[1]:].strip()])
            elif mat2:
                result.append([mat2[0][1:-1],title[mat2.span()[1]:].strip()])
    time.sleep(2)

cur_year_df = pd.DataFrame(result, columns =['Timestamp', 'Headline'])
cur_year_df.drop_duplicates(subset=['Headline'], inplace=True)
cur_year_df.to_csv(f"talk.csv",index=False)


