import numpy as np
import csv
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import math
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 하이퍼파라미터 설정
input_size = (160, 160)
batch_size = 32
num_classes = 528
initial_lr = 0.001
fine_tune_lr = 0.0001
epochs_initial = 20
epochs_finetune = 10

# 학습률 스케줄러
def step_decay(epoch):
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

# 데이터 경로 설정
train_dataset_dir = 'train/'
val_dataset_dir = 'val/'

# ImageDataGenerator로 데이터 로드 (증강 제거)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dataset_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    val_dataset_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# FaceNet 모델 로드 및 추가 레이어 정의
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')

# 기존 레이어 대신 Flatten 사용
x = base_model.output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# Base model은 frozen하여 전이 학습
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=initial_lr), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

lr_scheduler = LearningRateScheduler(step_decay)

callbacks = [checkpoint, early_stopping, reduce_lr, lr_scheduler]

# 모델 학습 (초기 전이 학습 단계)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_initial,
    callbacks=callbacks
)

# 베이스 모델의 상위 레이어 풀기 (상위 40%만 학습)
for layer in base_model.layers[-int(len(base_model.layers) * 0.4):]:
    layer.trainable = True

# 학습률을 낮춰서 다시 컴파일
model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning 단계 학습
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    callbacks=callbacks
)

# 최종 모델 저장
model.save('final_model.h5')

# -------------------------------
# 성능 평가 코드 추가 및 파일 저장
# -------------------------------

# 검증 데이터에 대한 예측
y_true = val_generator.classes  # 실제 레이블
y_pred = model.predict(val_generator, steps=val_generator.samples // batch_size + 1)
y_pred_classes = np.argmax(y_pred, axis=1)  # 예측된 레이블

# 1. 정확도 (Accuracy)
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Accuracy: {accuracy:.4f}')

# 2. 정밀도, 재현율, F1 Score
classification_rep = classification_report(y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys()))
print("Classification Report:")
print(classification_rep)

# 3. 혼동 행렬 (Confusion Matrix)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 4. Top-5 Accuracy
top_5_accuracy = top_k_accuracy_score(y_true, y_pred, k=5)
print(f'Top-5 Accuracy: {top_5_accuracy:.4f}')

# 성능 평가 지표 요약
performance_summary = {
    "Accuracy": accuracy,
    "Top-5 Accuracy": top_5_accuracy
}

# -------------------------------
# 1. 텍스트 파일로 저장
# -------------------------------
with open('model_performance_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_rep)
    f.write("\n")
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Top-5 Accuracy: {top_5_accuracy:.4f}\n')

# -------------------------------
# 2. CSV 파일로 저장
# -------------------------------
csv_file = 'model_performance_summary.csv'
csv_columns = ['Metric', 'Value']

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)
    for key, value in performance_summary.items():
        writer.writerow([key, value])

print(f"Performance results saved to 'model_performance_report.txt' and '{csv_file}'")

# -------------------------------
# 성능 평가 그래프 저장 및 파일 출력 추가
# -------------------------------

# 학습 정확도 및 손실 그래프 저장 함수
def plot_performance_and_save(history, fine_tune_history=None, save_path='accuracy_loss_plot.png'):
    plt.figure(figsize=(12, 5))

    # 1. 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if fine_tune_history:
        plt.plot(
            [i + len(history.history['accuracy']) for i in range(len(fine_tune_history.history['accuracy']))],
            fine_tune_history.history['accuracy'],
            label='Fine-tune Training Accuracy'
        )
        plt.plot(
            [i + len(history.history['val_accuracy']) for i in range(len(fine_tune_history.history['val_accuracy']))],
            fine_tune_history.history['val_accuracy'],
            label='Fine-tune Validation Accuracy'
        )
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 2. 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if fine_tune_history:
        plt.plot(
            [i + len(history.history['loss']) for i in range(len(fine_tune_history.history['loss']))],
            fine_tune_history.history['loss'],
            label='Fine-tune Training Loss'
        )
        plt.plot(
            [i + len(history.history['val_loss']) for i in range(len(fine_tune_history.history['val_loss']))],
            fine_tune_history.history['val_loss'],
            label='Fine-tune Validation Loss'
        )
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # 그래프를 파일로 저장
    plt.savefig(save_path)
    print(f"Performance plot saved as '{save_path}'")

# 학습 및 평가 성능 그래프 파일로 저장
plot_performance_and_save(history, fine_tune_history=history_finetune, save_path='accuracy_loss_plot.png')
