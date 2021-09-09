import requests
from  bs4 import BeautifulSoup
import pandas as pd
import datetime


# url = ('https://www.worldometers.info/coronavirus/country/india')
# r = requests.get(url)
# web_content = BeautifulSoup(r.text, 'lxml')
# web_content = web_content.find('div', {"class": 'maincounter-number'})
# web_content = web_content.find('span').text
#
# print(web_content)
#
# def real_time_data():
#     header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.75' }
#     url = f'https://www.worldometers.info/coronavirus/country/india'
#     r = requests.get(url)
#
#     soup = BeautifulSoup(r.text,'html.parser')
#
#     numbers ={'number':soup.find('div',{'class':'maincounter-number'}).find_all('span')[0].text}
#
#     return numbers
#
# print(real_time_data())

################################################################working model############################################
def real_time_data():
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.75'}
    url = ('https://www.worldometers.info/coronavirus/country/india')
    r = requests.get(url)
#   web_content = BeautifulSoup(r.text,'lxml')
    web_content =BeautifulSoup(r.text,'html.parser')
    web_content = web_content.find('div',{'class':'maincounter-number'}).find_all('span')[0].text
#   web_content = web_content.find('div',{"class":'maincounter-number'})
#   web_content= web_content.find('span').text

    # return web_content

# print(real_time_data())
#
#
    if web_content == []:
        web_content ='99999'

    return web_content

index = ['0001','0002','0003','0005']


for step in range(1,101):
    corona_no = []
    col = []
    time_stamp = datetime.datetime.now()
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M:%S")

    for corona in index :
        corona_no.append(real_time_data())
    col = [time_stamp]
    col.extend(corona_no)
    df =pd.DataFrame(col)
    df=df.T
    df.to_csv('real time corona death in India.csv',mode='a',header= False)
    print(col)
