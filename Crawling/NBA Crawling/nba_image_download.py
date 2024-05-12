import urllib.request
import urllib.error
import os
from nba_api.stats.static import players

player_list = players.get_players()

indexes_to_delete = []

# 원하는 디렉토리 경로 지정
desired_directory = 'Crawling/NBA Crawling'
path = os.path.join(desired_directory, 'Player Standard Data')

# 경로가 존재하지 않으면 생성
if not os.path.exists(path):
    os.makedirs(path)

for i in range(len(player_list)):
    if not player_list[i]['is_active']:
        indexes_to_delete.append(i)

indexes_to_delete.reverse()
for i in indexes_to_delete:
    del player_list[i]

for player in player_list:
    id = player['id']
    name = player['full_name']

    try:
        # 파일 경로에서 역슬래시를 슬래시로 변경
        urllib.request.urlretrieve(f'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{id}.png', f'{path}/{name}_std.jpg')
    except:
        print(f'{name} 이미지 다운로드 실패')