from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import urllib.request
import os

from nba_api.stats.static import players

player_list = players.get_players()

# 비활성 선수 제거
player_list = [player for player in player_list if player['is_active']]

# 이름 수정 함수
def modify_string(input_string):
    modified_string = input_string.replace(" ", "-")
    modified_string = modified_string.lower()
    return modified_string

# 선수 이름 수정
for player in player_list:
    player['full_name'] = modify_string(player['full_name'])

# 원하는 디렉토리 경로 지정
desired_directory = 'Crawling/Video Crawling/NBA Videos'

# 경로가 존재하지 않으면 생성
if not os.path.exists(desired_directory):
    os.makedirs(desired_directory)

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

# WebDriver 초기화
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 웹페이지 이동
driver.get("https://www.nba.com/players")

# 대기
driver.implicitly_wait(10)

for player in player_list:
    player_name = player['full_name']
    player_id = player['id']

    # 선수별 디렉토리 생성
    player_directory = os.path.join(desired_directory, player_name)
    if not os.path.exists(player_directory):
        os.makedirs(player_directory)

    # 선수 페이지로 이동
    driver.get(f'https://www.nba.com/player/{player_id}/{player_name}')

    # 쿠키 팝업 닫기
    try:
        cookie_ignore = driver.find_element(By.XPATH, "//*[starts-with(@class, 'onetrust-close-btn-handler banner-close-button ot-close-link')]")
        cookie_ignore.click()
    except:
        pass

    # 비디오 링크 찾기
    videos = driver.find_elements(By.XPATH, "//*[starts-with(@class, 'VideoSlide_image__3E8E9')]")
    count = len(videos)

    for i in range(count):
        driver.execute_script("window.scrollTo(0, 500)")

        videos_2 = driver.find_elements(By.XPATH, "//*[starts-with(@class, 'VideoSlide_image__3E8E9')]")

        try:
            videos_2[i].click()
        except:
            try:
                button = driver.find_element(By.XPATH, "//*[starts-with(@class, 'CarouselDynamic_next__5i0Dr w-10 h-10 CarouselButton_btnFloating__7MI1F')]")
                button.click()
                videos_2[i].click()
            except:
                button = driver.find_element(By.XPATH, "//*[starts-with(@class, 'CarouselDynamic_next__5i0Dr w-10 h-10 CarouselButton_btnFloating__7MI1F')]")
                button.click()
                videos_2[i].click()

        driver.implicitly_wait(5)

        # 비디오 링크 가져오기
        video_element = driver.find_element(By.XPATH, "//*[starts-with(@class, 'vjs-tech')]")
        video_src = video_element.get_attribute('src')

        # 비디오 저장
        try:
            urllib.request.urlretrieve(video_src, os.path.join(player_directory, f'{player_name}_video_{i}.mp4'))
        except:
            print('error')
            pass

        driver.get(f'https://www.nba.com/player/{player_id}/{player_name}')
        driver.implicitly_wait(5)