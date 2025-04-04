import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

#  Dataset-stier
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")

#  Kombiner "NoFish" med "NoFish_augmented"
no_fish_dir = os.path.join(train_dir, "NoFish")
no_fish_aug_dir = os.path.join(train_dir, "NoFish_augmented")

#  Flytt bilder fra NoFish_augmented til NoFish
for file_name in os.listdir(no_fish_aug_dir):
    src_path = os.path.join(no_fish_aug_dir, file_name)
    dest_path = os.path.join(no_fish_dir, file_name)
    shutil.move(src_path, dest_path)  # Flytter bildet til hovedmappen

#  Juster batch size og bildebehandling
img_size = (224, 224)
batch_size = 16  # Økt batch size for å håndtere mer data

#  Dataaugmentering for å forbedre generalisering
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

#  Last inn treningsdata (nå fra én mappe per klasse)
train_generator = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)

print(" Dataset er oppdatert og balansert!")
