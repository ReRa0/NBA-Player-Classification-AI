import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import shutil
from nba_api.stats.static import players
import re
from mtcnn import MTCNN
from PIL import Image

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 얼굴 검출 함수
def detect_faces(image_path):
    img = Image.open(image_path).convert('RGB')  # RGB 모드로 변환
    img = np.array(img)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces

# 얼굴 저장 함수
def save_detected_faces(image_path, faces, output_dir):
    img = Image.open(image_path).convert('RGB')  # RGB 모드로 변환
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        face_img = img.crop((x, y, x + width, y + height)).convert('RGB')  # RGB 모드로 변환
        face_img.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i}.jpg"))

# Remove non-English characters from player names
def remove_non_english(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z]', '', input_string)
    return cleaned_string

# Get active NBA players
player_list_for_google = players.get_players()
player_list_for_nba = players.get_players()
indexes_to_delete = [i for i in range(len(player_list_for_google)) if not player_list_for_google[i]['is_active']]
for i in reversed(indexes_to_delete):
    del player_list_for_google[i]
    del player_list_for_nba[i]
for player in player_list_for_google:
    player['full_name'] = remove_non_english(player['full_name'])

for i in range(1,len(player_list_for_nba)):

    player_google = player_list_for_google[i]['full_name']
    player_nba = player_list_for_nba[i]['full_name']

    os.mkdir(f'Crawling/Google Crawling/Clustered Image/{player_nba}')

    # 이미지 디렉토리 설정
    image_dir = f'Crawling/Google Crawling/Google Player Image Data/{player_google}/'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # 특정 이미지 경로
    specific_image_path = f'Crawling/NBA Crawling/Player Standard Data/{player_nba}_std.jpg'

    # VGG16 모델 불러오기
    model = VGG16(weights='imagenet', include_top=False)

    # 특징 추출
    features = []
    image_paths = []

    # 특정 이미지를 포함시키기
    specific_img_array = load_and_preprocess_image(specific_image_path)
    specific_feature = model.predict(specific_img_array).flatten()
    features.append(specific_feature)
    image_paths.append(specific_image_path)

    for img_path in image_files:
        img_array = load_and_preprocess_image(img_path)
        feature = model.predict(img_array)
        features.append(feature.flatten())
        image_paths.append(img_path)

    features = np.array(features)

    # PCA를 사용한 차원 축소
    n_components = min(len(features), features.shape[1])
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    # K-means 클러스터링
    n_clusters = 9
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced_features)
    labels = kmeans.labels_

    # 특정 이미지의 군집 확인
    specific_image_cluster = labels[0]

    # t-SNE를 사용한 2D 시각화
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(reduced_features)

    plt.figure(figsize=(8, 8))
    for i in range(n_clusters):
        indices = labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Cluster {i}')
    plt.legend()
    plt.savefig(f'Crawling/Google Crawling/Clustered Image/{player_nba}/{player_nba}_statistical_data.png')
    plt.close()

    # 군집별 폴더에 이미지 저장
    output_dir = f'Crawling/Google Crawling/Clustered Image/{player_nba}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cluster_dirs = {}
    for cluster in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster}')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        cluster_dirs[cluster] = cluster_dir

    # 특정 이미지가 속한 군집의 폴더 이름 변경
    specific_cluster_dir = cluster_dirs[specific_image_cluster]
    new_specific_cluster_dir = os.path.join(output_dir, f'cluster_{specific_image_cluster}_specific_image')
    os.rename(specific_cluster_dir, new_specific_cluster_dir)
    cluster_dirs[specific_image_cluster] = new_specific_cluster_dir

    # 이미지 복사
    for img_path, label in zip(image_paths, labels):
        cluster_dir = cluster_dirs[label]
        shutil.copy(img_path, cluster_dir)

    # MTCNN을 이용한 얼굴 검출 및 저장
    specific_image_cluster_dir = os.path.join(output_dir, f'cluster_{specific_image_cluster}_specific_image')
    detected_faces_dir = os.path.join(specific_image_cluster_dir, 'detected_faces')
    if not os.path.exists(detected_faces_dir):
        os.makedirs(detected_faces_dir)

    for img_file in os.listdir(specific_image_cluster_dir):
        img_path = os.path.join(specific_image_cluster_dir, img_file)
        if img_path.endswith('.jpg'):
            faces = detect_faces(img_path)
            save_detected_faces(img_path, faces, detected_faces_dir)
