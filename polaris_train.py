import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = (128, 128)
CLASSES = ['wildfire', 'normal']
DATA_DIR = 'POLARIS_satellite_dataset'

# === LOAD DATA === 
def load_dataset():
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, label)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                img = load_img(path, target_size=IMG_SIZE)
                img = img_to_array(img) / 255.0
                X.append(img)
                y.append(idx)
            except Exception as e:
                print(f"Skipped {path}: {e}")
    return np.array(X), to_categorical(y)

# === BUILD MODEL ===
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === MAIN ===
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Loaded {len(X)} images.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building and training model...")
    model = build_model()
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # === PLOT ===
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('POLARIS Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
