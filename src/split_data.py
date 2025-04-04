
import os
import shutil
import random

# Definer stier
base_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "..", "AnadromSmall")

fish_path = os.path.join(data_dir, "Fish")
nofish_path = os.path.join(data_dir, "NoFish")

output_dir = os.path.join(base_dir, "..", "dataset")

# Prosentandel for splitting
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Opprett mapper hvis de ikke finnes
for category in ["train", "val", "test"]:
    for label in ["Fish", "NoFish"]:
        os.makedirs(os.path.join(output_dir, category, label), exist_ok=True)

def split_and_move_images(src_folder, label):
    images = os.listdir(src_folder)
    random.shuffle(images)  # Bland bildene tilfeldig

    num_train = int(len(images) * train_ratio)
    num_val = int(len(images) * val_ratio)

    train_files = images[:num_train]
    val_files = images[num_train:num_train + num_val]
    test_files = images[num_train + num_val:]

    for file, subset in zip([train_files, val_files, test_files], ["train", "val", "test"]):
        dest_folder = os.path.join(output_dir, subset, label)
        for img in file:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dest_folder, img))

    print(f" {label}: {len(train_files)} trening, {len(val_files)} validering, {len(test_files)} test")

# Del opp datasettene
split_and_move_images(fish_path, "Fish")
split_and_move_images(nofish_path, "NoFish")

print("\n Data er delt opp i trenings-, validerings- og testsett!")
