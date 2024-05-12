from pytube import YouTube
import requests
from bs4 import BeautifulSoup


url = "https://www.youtube.com/@NBA/videos"
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

# href 가져오기
hrefs = []
for link in soup.find_all('a'):
    href = link.get('href')
    if href.startswith('/watch'):
        hrefs.append(href)

base_url = "https://www.youtube.com"
absolute_hrefs = [base_url + href for href in hrefs]

for href in absolute_hrefs:
    print(href)

exit()
  
# 저장할 경로 설정
SAVE_PATH = "video" # 경로 설정 
  
# 다운로드할 유튜브 링크 리스트에 담기
link=["https://www.youtube.com/watch?v=xWOoBJUqlbI", 
    "https://www.youtube.com/watch?v=xWOoBJUqlbI"
    ]
 
# for문 돌려서 다운로드받기
for i in link: 
    try:    
        # Youtube객채 생성
        yt = YouTube(i) 
    except: 
          
        # 예외처리
        print("Connection Error") 
      
    # 모든 파일 mp4 저장으로 설정
    mp4files = yt.filter('mp4') 
  
    # get()메서드로 해상도, 비디오 확장자 받기
    d_video = yt.get(mp4files[-1].extension,mp4files[-1].resolution) 
    try: 
        # 비디오 다운받기
        d_video.download(SAVE_PATH) 
    except: 
        print("에러 발생!") 
print('다운 완료!')