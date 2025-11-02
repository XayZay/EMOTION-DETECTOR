"""
Face Emotion Detection - Model Training Script
This script trains a CNN model to recognize 7 facial emotions.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = 'fer2013'  # Path to the dataset
IMG_SIZE = 48  # Images are 48x48 pixels
BATCH_SIZE = 32
EPOCHS = 50  # Maximum epochs (early stopping will prevent overfitting)
NUM_CLASSES = 7  # 7 emotions

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print("=" * 60)
print("FACE EMOTION DETECTION - MODEL TRAINING")
print("=" * 60)
print(f"Dataset Path: {DATASET_PATH}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Epochs: {EPOCHS}")
print(f"Emotions: {EMOTIONS}")
print("=" * 60)

# ============================================
# DATA LOADING AND AUGMENTATION
# ============================================
print("\n[1/5] Loading and preparing dataset...")

# Data augmentation for training (creates variations to prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to 0-1
    rotation_range=10,  # Randomly rotate images by 10 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally
    height_shift_range=0.1,  # Randomly shift images vertically
    horizontal_flip=True,  # Randomly flip images
    zoom_range=0.1,  # Randomly zoom
    fill_mode='nearest'
)

# No augmentation for test data (only normalization)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")
print(f"✓ Classes detected: {train_generator.class_indices}")

# ============================================
# MODEL ARCHITECTURE
# ============================================
print("\n[2/5] Building CNN model...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # First Convolutional Block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Fourth Convolutional Block
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output layer (7 emotions)
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built successfully!")
model.summary()

# ============================================
# CALLBACKS (Training helpers)
# ============================================
print("\n[3/5] Setting up training callbacks...")

# Stop training if validation loss doesn't improve for 10 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]
print("✓ Callbacks configured!")

# ============================================
# MODEL TRAINING
# ============================================
print("\n[4/5] Starting model training...")
print("⏳ This will take 30-60 minutes depending on your hardware...")
print("-" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed!")

# ============================================
# MODEL EVALUATION
# ============================================
print("\n[5/5] Evaluating model...")

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================
# SAVE MODEL
# ============================================
print("\nSaving model...")
model.save('face_emotionModel.h5')
print("✓ Model saved as 'face_emotionModel.h5'")

# ============================================
# PLOT TRAINING HISTORY
# ============================================
print("\nGenerating training plots...")

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✓ Training plots saved as 'training_history.png'")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("Model file: face_emotionModel.h5")
print("You can now proceed to Step 4: Building the Flask Web App")
print("=" * 60)