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

def scroll_down():
    while True:
        time.sleep(3)
        # 페이지 맨 아래로 스크롤
        driver.find_element(By.XPATH, '//body').send_keys(Keys.END)
        time.sleep(1)
        try:
            # '더보기' 버튼이 보이면 클릭
            load_more_button = driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
            if load_more_button.is_displayed():
                load_more_button.click()
        except:
            pass
        time.sleep(1)
        try:
            # '더 이상 표시할 콘텐츠가 없습니다.' 메시지가 보이면 종료
            no_more_content = driver.find_element(By.XPATH, '//div[@class="K25wae"]//*[text()="더 이상 표시할 콘텐츠가 없습니다."]')
            if no_more_content.is_displayed():
                break
        except:
            pass


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
        if image_url:
            image_path = os.path.join(path, f"{tag}_{idx}.jpg")
            urllib.request.urlretrieve(image_url, image_path)

driver.quit()
