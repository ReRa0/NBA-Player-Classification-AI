import numpy as np
import csv
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import math
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters
input_size = (224, 224)
batch_size = 64
num_classes = 1  # One class for "LeBron James"
initial_lr = 0.001
fine_tune_lr = 0.0001
epochs_initial = 20
epochs_finetune = 10

# Learning rate scheduler
def step_decay(epoch):
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

# Data paths
train_dataset_dir = 'train/'
val_dataset_dir = 'val/'

# Load data with ImageDataGenerator (no augmentation)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dataset_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary'  # Binary for single class
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    val_dataset_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Load base model and add custom layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation='sigmoid')(x)  # Sigmoid for binary classification

# Define model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=initial_lr), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
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

# Initial training phase
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_initial,
    callbacks=callbacks
)

# Unfreeze top layers for fine-tuning
for layer in base_model.layers[-int(len(base_model.layers) * 0.4):]:
    layer.trainable = True

# Recompile model with lower learning rate
model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tuning phase
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    callbacks=callbacks
)

# Save final model
model.save('final_model.h5')

# --------------------------------
# Performance evaluation and save
# --------------------------------

# Predict on validation set
y_true = val_generator.classes  # Actual labels
y_pred = model.predict(val_generator, steps=val_generator.samples // batch_size + 1)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # Binary classification threshold

# 1. Accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Accuracy: {accuracy:.4f}')

# 2. Classification report
classification_rep = classification_report(y_true, y_pred_classes, target_names=["LeBron James"])
print("Classification Report:")
print(classification_rep)

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["LeBron James"], yticklabels=["LeBron James"])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Performance summary dictionary
performance_summary = {
    "Accuracy": accuracy
}

# --------------------------------
# 1. Save as text file
# --------------------------------
with open('model_performance_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_rep)
    f.write("\n")
    f.write(f'Accuracy: {accuracy:.4f}\n')

# --------------------------------
# 2. Save as CSV file
# --------------------------------
csv_file = 'model_performance_summary.csv'
csv_columns = ['Metric', 'Value']
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)
    for key, value in performance_summary.items():
        writer.writerow([key, value])

print(f"Performance results saved to 'model_performance_report.txt' and '{csv_file}'")

# --------------------------------
# Save performance graph
# --------------------------------

def plot_performance_and_save(history, fine_tune_history=None, save_path='accuracy_loss_plot.png'):
    plt.figure(figsize=(12, 5))

    # 1. Accuracy plot
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

    # 2. Loss plot
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
    plt.savefig(save_path)
    print(f"Performance plot saved as '{save_path}'")

plot_performance_and_save(history, fine_tune_history=history_finetune, save_path='accuracy_loss_plot.png')
