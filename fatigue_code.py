import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set the path to your dataset
data_dir = "C:/Users/rumaa/OneDrive/Documents/train"

# Define image dimensions and other hyperparameters
img_width, img_height = 100, 100
batch_size = 32
epochs = 10

# Data augmentation to prevent overfitting
data_augmentation = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess data using data augmentation
train_generator = data_augmentation.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  # Assumes 0 for closed eyes and 1 for opened eyes
    shuffle=True
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Save the trained model
model.save("C:/Users/rumaa/OneDrive/Documents/drowsiness_detection_model_new.h5")

# Load the model
loaded_model = tf.keras.models.load_model("C:/Users/rumaa/OneDrive/Documents/drowsiness_detection_model_new.h5")

# Now you can use the loaded model to make predictions on new data
# Example: Assuming 'new_image' is the new image you want to predict
new_image = tf.keras.preprocessing.image.load_img("C:/Users/rumaa/OneDrive/Documents/train/Closed_Eyes/s0001_00462_0_0_0_0_0_01.png",
                                                  target_size=(img_width, img_height))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)
prediction = loaded_model.predict(new_image)

if prediction < 0.5:
    print("Closed eyes - Person is likely sleepy.")
else:
    print("Opened eyes - Person is likely awake.")
