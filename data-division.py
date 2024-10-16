import os
import shutil
import random

# 경로 설정
src_dir = 'Crawling/Google Crawling/Augmentation Image/'  # 원본 데이터 폴더
dst_base_dir = 'D:/NBA-Player-Classification-AI/'  # 새로운 데이터 저장할 위치
train_dir = os.path.join(dst_base_dir, 'train')  # 학습 데이터 폴더
val_dir = os.path.join(dst_base_dir, 'val')  # 검증 데이터 폴더

# 학습과 검증 데이터를 나누는 비율 (80% 학습, 20% 검증)
train_ratio = 0.8

# 폴더가 없으면 생성
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 클래스 디렉토리 리스트
class_names = os.listdir(src_dir)

for class_name in class_names:
    class_src_path = os.path.join(src_dir, class_name)
    
    # 클래스별 이미지 파일 리스트
    images = os.listdir(class_src_path)
    random.shuffle(images)  # 랜덤하게 섞기
    
    # 학습과 검증 데이터로 분리
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # 학습 데이터 저장 경로
    train_class_dir = os.path.join(train_dir, class_name)
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    
    # 검증 데이터 저장 경로
    val_class_dir = os.path.join(val_dir, class_name)
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)
    
    # 파일 복사 (학습 데이터)
    for img in train_images:
        shutil.copy(os.path.join(class_src_path, img), os.path.join(train_class_dir, img))
    
    # 파일 복사 (검증 데이터)
    for img in val_images:
        shutil.copy(os.path.join(class_src_path, img), os.path.join(val_class_dir, img))

print("데이터 분리 완료.")
