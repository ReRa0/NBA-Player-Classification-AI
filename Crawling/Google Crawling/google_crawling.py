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

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.google.com/")

for i in range(len(player_list)):

    tag = player_list[i]['full_name']
    url = f'https://www.google.com/search?tbm=isch&q={tag}'

    os.makedirs(f'Crawling/Google Crawling/Google Player Image Data/{tag}', exist_ok=True)
    path = f'Crawling/Google Crawling/Google Player Image Data/{tag}'

    driver.get(url)

    time.sleep(2)

    driver.implicitly_wait(15)

    images = driver.find_elements(By.CSS_SELECTOR, '.rg_i.Q4LuWd')
    
    for idx, image in enumerate(images):
        image_url = image.get_attribute('src')
        if image_url:
            image_path = os.path.join(path, f"{tag}_{idx}.jpg")
            urllib.request.urlretrieve(image_url, image_path)

driver.quit()
