import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# === CONFIG ===
img_size = 224
batch_size = 16
train_dir = 'dataset/train'
val_dir = 'dataset/val'
epochs = 25

# === DATA LOADERS ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# === CLASS WEIGHTS ===
labels = train_gen.classes  # numpy array of 0s and 1s
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Computed class weights:", class_weight_dict)

# === MODEL ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === TRAIN ===
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weight_dict
)

# === UNFREEZE AND FINE-TUNE (optional) ===
# for layer in base_model.layers:
#     layer.trainable = True
# model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_gen, validation_data=val_gen, epochs=5)

# === SAVE ===
model.save("brick_defect_model.h5")
print("✅ Model training complete and saved.")
