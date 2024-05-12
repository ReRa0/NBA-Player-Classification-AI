import urllib.request
import urllib.error

from nba_api.stats.static import players

player_list = players.get_players()

indexes_to_delete = []

for i in range(len(player_list)):
    if not player_list[i]['is_active']:
        indexes_to_delete.append(i)

indexes_to_delete.reverse()
for i in indexes_to_delete:
    del player_list[i]

for i in range(len(player_list)):

    id = player_list[i]['id']
    name = player_list[i]['full_name']

    try:
        urllib.request.urlretrieve(f'https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{id}.png', f'Player Standard Data\{name}_std.jpg')
    except:
        print(f'{name} 이미지 다운로드 실패')
