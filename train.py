import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Dataset Path
dataset_path = r"D:\ML\food\Food Images"

# Image Size
IMG_SIZE = (224, 224)

# Batch Size
BATCH_SIZE = 32

# Epochs
EPOCHS = 5

# Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training Data
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training'
)

# Validation Data
val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# Load MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze Base Model
base_model.trainable = False

# Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)

predictions = Dense(
    train_data.num_classes,
    activation='softmax'
)(x)

# Final Model
model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Create Models Folder
os.makedirs("models", exist_ok=True)

# Save Model
model.save("models/food_model.h5")

print("Model Saved Successfully!")