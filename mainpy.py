import requests
from bs4 import BeautifulSoup
import pandas as pd

''' Writing final output to a DF '''
list_alphabet = list(map(lambda i:i, 'abcdefghijklmnopqrstuvwxyz'))
df = []
links = []
colnames = ['First','Last','Nickname','Height','Weight','Reach','Stance','W','L','D']
df = pd.DataFrame(columns=colnames)

for letter in list_alphabet:    
    page = requests.get('http://www.ufcstats.com/statistics/fighters?char={}&page=all'.format(letter))
    soup = BeautifulSoup(page.content, 'html.parser')
    
    ''' Finding all hyperlinks'''
    for a in soup.find_all('a', href=True):
        if 'http://www.ufcstats.com/fighter-details/' in str(a['href']):
            if str(a['href']) not in links:
                links.append(a['href'])
    
    parsed_html = soup.select('tr td')
    fighter_row = []
    for i in range(1,len(parsed_html)):
        if i % 11 == 0:
            df = df.append(pd.DataFrame(data=[fighter_row],columns=colnames), ignore_index=True)
            fighter_row = []
            continue
        fighter_row.append(parsed_html[i].get_text().replace('\n',''))

df['Links'] = links