import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    print(" Starter dataaugmentering...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    print(" Laster treningsdata...")
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    
    print(" Laster valideringsdata...")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )

    print(" Laster testdata...")
    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )

    print(" Dataaugmentering klar!")
    return train_generator, val_generator, test_generator

# Debugging
if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators("dataset/train", "dataset/val", "dataset/test")
