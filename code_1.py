import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D

# Function to load dataset with preprocessing

def generator(dir, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=32, target_size=(24, 24), class_mode='categorical'):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )

# Parameters
BS = 32   # Batch size
TS = (24, 24)  # Target image size


# Training and validation batches
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

SPE = len(train_batch.classes) // BS   # steps_per_epoch
VS = len(valid_batch.classes) // BS    # validation steps

# Model Creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')   # assuming 2 classes: drowsy & alert
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_batch,
    steps_per_epoch=SPE,
    epochs=10,
    validation_data=valid_batch,
    validation_steps=VS,
    verbose=1
)

# Save the trained model
if not os.path.exists("saved_model"):
    os.mkdir("saved_model")

model.save("saved_model/drowsiness_model.h5")

# Plot Training Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
