import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# 이미지 경로 및 데이터 불러오기
train_dataset_dir = 'Crawling/Google Crawling/Augmentation Image'
val_dataset_dir = 'Crawling/Google Crawling/Google Player Image Data'

# ImageDataGenerator로 데이터 로드 (훈련 데이터)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)  # 이미지 정규화
train_generator = train_datagen.flow_from_directory(
    train_dataset_dir,
    target_size=(160, 160),  # FaceNet 모델의 입력 크기에 맞춤
    batch_size=32,
    class_mode='categorical'  # 폴더 이름을 레이블로
)

# 검증 데이터 제너레이터 설정
val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # 이미지 정규화
val_generator = val_datagen.flow_from_directory(
    val_dataset_dir,
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

# 모델 체크포인트 설정 (가장 좋은 검증 성능을 가진 모델을 저장)
checkpoint = ModelCheckpoint(
    'best_model.h5',  # 저장할 파일 이름
    monitor='val_accuracy',  # 모니터링할 성능 지표
    save_best_only=True,  # 최고 성능일 때만 저장
    mode='max',  # 최대화할 지표
    verbose=1  # 저장 시 메시지 출력
)

# 모델 학습
history = model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[checkpoint])

# Base model을 풀어 전체 학습
for layer in base_model.layers:
    layer.trainable = True

# 학습률을 낮춰서 다시 컴파일 후 학습
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning
history_finetune = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[checkpoint])

# 훈련이 끝났을 때 모델 저장
model.save('final_model.h5')
