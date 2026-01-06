import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples
import os

# Settingan biar gak berisik warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("--- MULAI PROSES TRAINING ULANG (FIXED VERSION) ---")
    print("1. Loading dataset EMNIST...")
    
   
    train_images, train_labels = extract_training_samples('balanced')
    test_images, test_labels = extract_test_samples('balanced')

    print(f"   Data didapat: {len(train_images)} gambar training.")

    print("2. Normalisasi Gambar (Tanpa Diputar)...")
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255

    num_classes = 47
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    print("3. Membangun Arsitektur CNN...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("4. Gas Training! (Tunggu bentar)...")
    history = model.fit(train_images, train_labels, 
                        epochs=5, 
                        batch_size=64,
                        validation_data=(test_images, test_labels))

    print("5. Menyimpan hasil...")
    model.save('handwriting_model.h5')
    print("   -> Model baru disimpan: handwriting_model.h5")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Akurasi Training')
    plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
    plt.title('Grafik Akurasi Model CNN (Fixed)')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    plt.grid(True)
    plt.savefig('laporan_akurasi_fixed.png')
    print("   -> Grafik baru disimpan: laporan_akurasi_fixed.png")
    print("--- SELESAI! REFRESH WEB APP LU ---")

if __name__ == "__main__":
    train()
