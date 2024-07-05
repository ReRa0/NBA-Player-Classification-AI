from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

import urllib.request
import ssl
import os
import re
import time

from nba_api.stats.static import players

# Remove non-English characters from player names
def remove_non_english(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z]', '', input_string)
    return cleaned_string

# Get active NBA players
player_list = players.get_players()
indexes_to_delete = [i for i in range(len(player_list)) if not player_list[i]['is_active']]
for i in reversed(indexes_to_delete):
    del player_list[i]
for player in player_list:
    player['full_name'] = remove_non_english(player['full_name'])

# Setup SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Setup Chrome driver options
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
service = Service(ChromeDriverManager().install())

# Initialize Chrome driver
driver = webdriver.Chrome(options=chrome_options)

# Function to scroll down the page
def scroll_down():
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.find_element(By.XPATH, '//body').send_keys(Keys.END)
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                load_more_button = driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
                if load_more_button.is_displayed():
                    load_more_button.click()
                    time.sleep(1)
                else:
                    break
            except:
                break
        last_height = new_height

# Crawl images for each player
for i in range(len(player_list)):
    tag = player_list[i]['full_name']
    url = f'https://www.google.com/search?tbm=isch&q={tag}'

    os.makedirs(f'Crawling/Google Crawling/Google Player Image Data/{tag}', exist_ok=True)
    path = f'Crawling/Google Crawling/Google Player Image Data/{tag}'

    driver.get(url)
    driver.implicitly_wait(15)

    scroll_down()

    images = driver.find_elements(By.CSS_SELECTOR, '.rg_i.Q4LuWd')
    
    for idx, image in enumerate(images):
        image_url = image.get_attribute('src')
        if not image_url:
            image_url = image.get_attribute('data-src')
        if not image_url:
            image_url = image.get_attribute('srcset').split(' ')[0] if image.get_attribute('srcset') else None
        
        if image_url:
            try:
                image_path = os.path.join(path, f"{tag}_{idx}.jpg")
                urllib.request.urlretrieve(image_url, image_path)
            except Exception as e:
                print(f"Could not download {image_url}: {e}")

driver.quit()
