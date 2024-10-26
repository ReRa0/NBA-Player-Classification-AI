import os
import shutil
import random

# 경로 설정
src_dir = r'C:\Users\adamk\Desktop\Image-2\LeBronJames'  # 원본 데이터 폴더
dst_base_dir = r'C:\Users\adamk\Desktop\NBA-Player-Classification-AI'  # 새로운 데이터 저장할 위치
train_dir = os.path.join(dst_base_dir, 'train')  # 학습 데이터 폴더
val_dir = os.path.join(dst_base_dir, 'val')  # 검증 데이터 폴더

# 학습과 검증 데이터를 나누는 비율 (80% 학습, 20% 검증)
train_ratio = 0.8

# 폴더가 없으면 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 이미지 파일 리스트 (.jpg, .png 등의 확장자 필터링)
images = [img for img in os.listdir(src_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

# 랜덤 시드 설정 (선택 사항)
random.seed(42)  # 시드를 제거하면 매번 다른 결과를 얻을 수 있습니다

# 랜덤하게 섞기
random.shuffle(images)

# 섞인 결과 확인
print("섞인 이미지 리스트 (일부):", images[:10])  # 처음 10개 출력

# 학습과 검증 데이터로 분리
split_index = int(len(images) * train_ratio)
train_images = images[:split_index]
val_images = images[split_index:]

# 파일 복사 (학습 데이터)
for img in train_images:
    shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, img))

# 파일 복사 (검증 데이터)
for img in val_images:
    shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, img))

print("데이터 분리 완료.")
print(f"총 이미지 수: {len(images)}")
print(f"학습 데이터 수: {len(train_images)}")
print(f"검증 데이터 수: {len(val_images)}")
