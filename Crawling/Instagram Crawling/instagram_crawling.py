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

from instagram_info import id, password

player_list = players.get_players()

def remove_non_english(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z]', '', input_string)
    return cleaned_string

indexes_to_delete = []

for i in range(len(player_list)):
    if not player_list[i]['is_active']:
        indexes_to_delete.append(i)

indexes_to_delete.reverse()
for i in indexes_to_delete:
    del player_list[i]

for i in range(len(player_list)):
    player_list[i]['full_name'] = remove_non_english(player_list[i]['full_name'])


ssl._create_default_https_context = ssl._create_unverified_context

# 브라우저 꺼짐 방지
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 불필요한 에러 메시지 없애기
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(options=chrome_options)

# 웹페이지 해당 주소 이동
driver.get("https://www.instagram.com/")

# 인스타그램 자동 로그인
driver.implicitly_wait(10) #나타날 때까지 최대 10초동안 기다림
login_id = driver.find_element(By.CSS_SELECTOR, "input[name='username']")
login_id.send_keys(id) #인스타그램 아이디
login_pwd = driver.find_element(By.CSS_SELECTOR, "input[name='password']")
login_pwd.send_keys(password) #인스타그램 비밀번호
driver.implicitly_wait(10)
login_id.send_keys(Keys.ENTER)

time.sleep(5)

for i in range(len(player_list)):

    tag = player_list[i]['full_name']
    url = f'https://www.instagram.com/explore/tags/{tag}/'

    os.makedirs(f'Player Image Data\{tag}', exist_ok=True)
    path = f'Player Image Data\{tag}'

    driver.get(url)

    driver.implicitly_wait(15)

    img = driver.find_elements(By.CSS_SELECTOR, 'div._aagv > img')
    for i in range(len(img)):
        urllib.request.urlretrieve(img[i].get_attribute('src'), f'{path}\{tag}{i}.jpg')