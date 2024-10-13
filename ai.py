import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

# 이미지 경로 및 데이터 불러오기
dataset_dir = 'Crawling/Google Crawling/Augmentation Image'

# ImageDataGenerator로 데이터 로드
datagen = ImageDataGenerator(rescale=1.0/255.0)  # 이미지 정규화

# 각 폴더의 이미지를 불러오기 (폴더 이름이 레이블로 사용됨)
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(160, 160),  # FaceNet 모델의 입력 크기에 맞춤
    batch_size=32,
    class_mode='categorical'  # 폴더 이름을 레이블로
)

# FaceNet 모델 로드
base_model = load_model('model/facenet_keras.h5')

# 새로운 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 평균 풀링 레이어
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(528, activation='softmax')(x)  # 528명의 클래스를 분류

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# Base model은 frozen하여 전이 학습
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(train_generator, epochs=20)

# Base model을 풀어 전체 학습
for layer in base_model.layers:
    layer.trainable = True

# 학습률을 낮춰서 다시 컴파일 후 학습
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning
history_finetune = model.fit(train_generator, epochs=10)
