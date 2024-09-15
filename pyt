# Proyek Jaringan Saraf Tiruan: Mengenali Bentuk Tangan (Gunting, Batu, Kertas)

# Nama: Muhammad Farhat Rafsanjani  
# Email: rafirafsanjani394@gmail.com
# Tanggal: 15 september  

!wget https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip
!unzip rockpaperscissors.zip -d rockpaperscissors

import os
import shutil
from sklearn.model_selection import train_test_split

# Direktori dataset
base_dir = 'rockpaperscissors/rockpaperscissors'
train_dir = 'rockpaperscissors/train'
val_dir = 'rockpaperscissors/validation'

# Membuat direktori untuk train dan validation
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Daftar subdirektori untuk masing-masing kelas
classes = ['rock', 'paper', 'scissors']

# Membagi data
for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    images = os.listdir(cls_dir)
    train_images, val_images = train_test_split(images, test_size=0.4, random_state=42)
    
    # Pindahkan gambar ke direktori train dan validation
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    
    for img in train_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
    
    for img in val_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentasi dan normalisasi gambar untuk pelatihan
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi pixel gambar
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normalisasi gambar untuk validasi
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generator untuk pelatihan
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Ukuran gambar
    batch_size=32,
    class_mode='categorical'
)

# Generator untuk validasi
validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
from tensorflow.keras import layers, models

# Membuat model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 kelas: rock, paper, scissors
])

# Mengkompilasi model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
# Evaluasi model pada data validasi
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {validation_accuracy:.4f}')
from tensorflow.keras.preprocessing import image
import numpy as np

# Fungsi untuk memuat dan memprediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    classes = ['rock', 'paper', 'scissors']
    predicted_class = classes[np.argmax(predictions)]
    
    return predicted_class

# Unggah gambar untuk diuji
from google.colab import files
uploaded = files.upload()

# Prediksi gambar yang diunggah
for img_name in uploaded.keys():
    result = predict_image(img_name)
    print(f'Gambar {img_name} diprediksi sebagai: {result}')

