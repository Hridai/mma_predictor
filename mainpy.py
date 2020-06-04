import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

def _remove_surrounding_whitespace(s):
    return s.strip()

def _replace(s, str_from, str_to):
    return s.replace(str_from, str_to)

def _feet_to_cm(s):
    footinch = s.split("'")
    if footinch[0] == s: return s # is a NA row
    cm_total = float(footinch[0]) * 12 * 2.54
    cm_total += float(footinch[1].strip()) * 2.54
    return cm_total

#''' Writing final output to a DF '''
#list_alphabet = list(map(lambda i:i, 'abcdefghijklmnopqrstuvwxyz'))
#colnames = ['First','Last','Nickname','Height','Weight','Reach','Stance','W','L','D']
#df = pd.DataFrame(columns=colnames)]

#links = []

#
#for letter in list_alphabet:    
#    page = requests.get('http://www.ufcstats.com/statistics/fighters?char={}&page=all'.format(letter))
#    soup = BeautifulSoup(page.content, 'html.parser')
#    
#    ''' Finding all hyperlinks'''
#    for a in soup.find_all('a', href=True):
#        if 'http://www.ufcstats.com/fighter-details/' in str(a['href']):
#            if str(a['href']) not in links:
#                links.append(a['href'])
#    
#    parsed_html = soup.select('tr td')
#    fighter_row = []
#    for i in range(1,len(parsed_html)):
#        if i % 11 == 0:
#            df = df.append(pd.DataFrame(data=[fighter_row],columns=colnames), ignore_index=True)
#            fighter_row = []
#            continue
#        fighter_row.append(parsed_html[i].get_text().replace('\n',''))
#
#df['Links'] = links

#''' post-process - data cleanse '''
#for col in colnames:
#    df[col] = df.apply(lambda x: _remove_surrounding_whitespace(x[col]), axis=1)
#df['Weight'] = df.apply(lambda x: _replace(x['Weight'],' lbs.',''), axis=1)
#df['Reach'] = df.apply(lambda x: _replace(x['Reach'],'"',''), axis=1)
#df['Height'] = df.apply(lambda x: _replace(x['Height'],'"',''), axis=1)
#df['Height'] = df.apply(lambda x: _feet_to_cm(x['Height']), axis=1)

''' note this is only to start from a loaded file '''
df_read = pd.read_excel('sub list.xlsx')
list_links = df_read['Links']
''' End File Load'''

colnames_bouts = ['Fighter1Result','Fighter1','Fighter2','F1Strikes','F2Strikes','F1TakeDowns','F2TakeDowns',
                  'F1Subs','F2Subs','F1Pass','F2Pass','Event','EventDate','VictoryMethod','VictoryDetail',
                  'Round','Time']
df_bouts = pd.DataFrame(columns=colnames_bouts)
colnames_2 = ['Link','DOB','SLpM','Str.Acc','SApM','Str.Def','TD Avg.','TD Acc.','TD Def.','Sub. Avg']
df_2 = pd.DataFrame(columns=colnames_2)

res_set_skip = ['m-dec','s-dec','u-dec','dq','overturned','decision','other','cnc','sub']
counter = 0
for link in list_links:
    print(f'{counter} processed')
#    time.sleep(4)
    counter += 1
    if counter % 100 == 0:
        df_2.to_excel('df2.xlsx')
        df_bouts.to_excel('bouts.xlsx')

    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    result_set = soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
    
    enhanced_fighter_row=[]
    enhanced_fighter_row.append(link)
    fields_keep = ['DOB:','SLpM:','Str. Acc.:','SApM:','Str. Def:','TD Avg.:','TD Acc.:','TD Def.:','Sub. Avg.:']
    for item in result_set:
        datapoint = item.get_text().replace('\n','').strip()
        data_split = re.split(r'\s{2,}', datapoint)
        if data_split[0] in fields_keep: enhanced_fighter_row.append(data_split[1])
    df_2 = df_2.append(pd.DataFrame(data=[enhanced_fighter_row],columns=colnames_2), ignore_index=True)
    
    bout_row = []
    result_list = ['win','loss','draw','nc']
    result_set_bout_stats = soup.find_all('td', class_='b-fight-details__table-col')
    bnextfound = False
    k = 0
    for item in result_set_bout_stats:    
        if k == 0:
            if str(item.get_text()).lower().replace('\n','') == 'next':
               bnextfound = True
        if bnextfound:
            if str(item.get_text()).lower().replace('\n','') in result_list:
                bnextfound = False
                bout_row = []
                k = 0
            else:
                k += 1
                continue
        detail = item.get_text().replace('\n','').strip()
        detail_split = re.split(r'\s{3,}', detail)
        for j, elem in enumerate(detail_split):
            bout_row.append(elem)
            if str(elem).lower() in res_set_skip:
                if j == 0:
                    if len(detail_split) < 2:
                        bout_row.append('')
                        break
            elif 'ko' in str(elem).lower():
                if len(detail_split) == 1:
                    bout_row.append('')
                    break
        k += 1

        if k == 10:
            df_bouts = df_bouts.append(pd.DataFrame(data=[bout_row],columns=colnames_bouts), ignore_index=True)
            bout_row = []
            k = 0
            
df_2.to_excel('df2.xlsx')
df_bouts.to_excel('bouts.xlsx')
print('...bish bosh bash...')