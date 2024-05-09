from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

import urllib.request
import urllib.error
import ssl
import os
import re
import time

from nba_api.stats.static import players
from instagram_crawling import data_length, new_player_list

ssl._create_default_https_context = ssl._create_unverified_context

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 불필요한 에러 메시지 없애기
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(options=chrome_options)

# 웹페이지 해당 주소 이동
driver.get("https://www.nba.com/players")

time.sleep(5)

for i in range(582):

    id = new_player_list[i]['id']
    name = new_player_list[i]['full_name']

    driver.get(f'https://www.nba.com/players/{id}/{name}')

    driver.implicitly_wait(15)

    os.makedirs(f'Player Standard Data\{name}', exist_ok=True)


    img = driver.find_elements(By.CSS_SELECTOR, 'PlayerSummary_mainInnerTeam____nFZ > img')
    urllib.request.urlretrieve(img[i].get_attribute('src'), f'{name}_std.jpg')