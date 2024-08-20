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
import cv2
from PIL import ImageFilter,Image, ImageEnhance,ImageOps
import io
import random

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 얼굴 검출 함수
def detect_faces(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces

# 얼굴 저장 함수
def save_detected_faces(image_path, faces, output_dir):
    img = Image.open(image_path).convert('RGB')
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        face_img = img.crop((x, y, x + width, y + height)).convert('RGB')
        face_img.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i}.jpg"))

# Remove non-English characters from player names
def remove_non_english(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z]', '', input_string)
    return cleaned_string

# 좌우 반전
def flip_image(image):
    return ImageOps.mirror(image)

# 회전
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

# 밝기 조절
def adjust_brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

# 채도 조절
def adjust_saturation(image, saturation_factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation_factor)

# 색조 조절
def adjust_hue(image, hue_factor):
    image = np.array(image.convert('HSV'))
    image[..., 0] = np.uint8((image[..., 0].astype(int) + hue_factor) % 180)
    image = Image.fromarray(image, mode='HSV').convert('RGB')
    return image

# 명도 조절
def adjust_contrast(image, contrast_factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)

# 블러
def apply_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_motion_blur(image, degree=15, angle=45):
    # Create an empty kernel
    kernel = np.zeros((degree, degree))
    
    # Fill the kernel to create a linear motion blur effect
    kernel[int((degree - 1)/2), :] = np.ones(degree)
    
    # Normalize the kernel
    kernel /= degree
    
    # Rotate the kernel to the specified angle using affine transformation
    (h, w) = kernel.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, M, (w, h))
    
    # Apply the kernel to the input image
    image = np.array(image)
    motion_blur_image = cv2.filter2D(image, -1, rotated_kernel)
    
    return Image.fromarray(np.uint8(motion_blur_image))

# 이미지 압축 (JPEG)
def apply_compression(image, quality=30):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    return Image.open(buffer)

# 노이즈 추가
def add_noise(image, noise_factor=0.05):
    image = np.array(image)
    noise = np.random.normal(0, noise_factor ** 0.5, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Augmentation 수행 함수 수정
def augment_images(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"입력 폴더가 존재하지 않습니다: {input_folder}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]
    if not image_files:
        print(f"입력 폴더에 이미지 파일이 없습니다: {input_folder}")
        return

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        with Image.open(image_path) as image:
            # 원본 이미지 저장
            image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_original.jpg"))

            # 좌우 반전
            flipped_image = flip_image(image)
            flipped_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_flipped.jpg"))

            # 회전 이미지 (3장)
            for i in range(3):
                angle = random.uniform(-15, 15)
                rotated_image = rotate_image(image, angle)
                rotated_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_rotated_{i}.jpg"))

            # 밝기 조절 이미지 (3장)
            for i in range(3):
                brightness_factor = random.uniform(0.5, 1.5)
                brightness_image = adjust_brightness(image, brightness_factor)
                brightness_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_brightness_{i}.jpg"))

            # 채도 조절 이미지 (3장)
            for i in range(3):
                saturation_factor = random.uniform(0.8, 1.2)
                saturation_image = adjust_saturation(image, saturation_factor)
                saturation_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_saturation_{i}.jpg"))

            # 색조 조절 이미지 (3장)
            for i in range(3):
                hue_factor = random.randint(-10, 10)
                hue_image = adjust_hue(image, hue_factor)
                hue_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_hue_{i}.jpg"))

            # 명도 조절 이미지 (3장)
            for i in range(3):
                contrast_factor = random.uniform(0.5, 1.5)
                contrast_image = adjust_contrast(image, contrast_factor)
                contrast_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_contrast_{i}.jpg"))

            # 블러 이미지 (2장)
            for i in range(2):
                blur_image = apply_blur(image)
                blur_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_blur_{i}.jpg"))

            # 모션 블러 이미지 (2장)
            for i in range(2):
                motion_blur_image = apply_motion_blur(image)
                motion_blur_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_motion_blur_{i}.jpg"))

            # 압축 이미지 (2장)
            for i in range(2):
                compressed_image = apply_compression(image)
                compressed_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_compression_{i}.jpg"))

            # 노이즈 추가 이미지 (2장)
            for i in range(2):
                noisy_image = add_noise(image)
                noisy_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_noise_{i}.jpg"))

            # 모든 효과를 합친 이미지
            for i in range(2):
                combined_image = adjust_brightness(image, random.uniform(0.5, 1.5))
                combined_image = adjust_saturation(combined_image, random.uniform(0.8, 1.2))
                combined_image = adjust_hue(combined_image, random.randint(-10, 10))
                combined_image = adjust_contrast(combined_image, random.uniform(0.5, 1.5))
                combined_image = apply_blur(combined_image)
                combined_image = apply_motion_blur(combined_image)
                combined_image = apply_compression(combined_image)
                combined_image = add_noise(combined_image)
                combined_image.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_combined_{i}.jpg"))

# Get active NBA players
player_list_for_google = players.get_players()
player_list_for_nba = players.get_players()
indexes_to_delete = [i for i in range(len(player_list_for_google)) if not player_list_for_google[i]['is_active']]
for i in reversed(indexes_to_delete):
    del player_list_for_google[i]
    del player_list_for_nba[i]
for player in player_list_for_google:
    player['full_name'] = remove_non_english(player['full_name'])

# 선수 데이터 루프 시작 (첫 번째 선수부터 시작)
for i in range(len(player_list_for_nba)):  # 첫 번째 선수부터 시작
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

    # MTCNN을 이용한 얼굴 검출 및 저장 (specific 이미지 폴더 내)
    specific_image_cluster_dir = os.path.join(output_dir, f'cluster_{specific_image_cluster}_specific_image')
    detected_faces_dir = os.path.join(specific_image_cluster_dir, 'detected_faces')
    if not os.path.exists(detected_faces_dir):
        os.makedirs(detected_faces_dir)

    for img_file in os.listdir(specific_image_cluster_dir):
        img_path = os.path.join(specific_image_cluster_dir, img_file)
        if img_path.endswith('.jpg'):
            faces = detect_faces(img_path)
            save_detected_faces(img_path, faces, detected_faces_dir)

    # 얼굴 이미지에 대해 Augmentation 수행 (specific 이미지 폴더 내)
    augment_images(detected_faces_dir, f"Crawling/Google Crawling/Augmentation Image/{player_nba}")

    print(f"{player_nba}에 대한 작업이 완료되었습니다.")
