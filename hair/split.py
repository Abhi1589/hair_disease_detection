import os
import random
import shutil

# Define the source folder and the target folders
source_folder = r"C:\Users\abhik\Downloads\archive (8)\data0330\bald"
train_folder = r"C:\Users\abhik\Downloads\Hair_Diseases\train\Normal_Hair"
test_folder = r"C:\Users\abhik\Downloads\Hair_Diseases\test\Normal_Hair"
val_folder = r"C:\Users\abhik\Downloads\Hair_Diseases\val\Normal_Hair"

# Make sure target folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get a list of all image files in the source folder
all_images = os.listdir(source_folder)

# Shuffle the list to ensure randomness
random.shuffle(all_images)

# Calculate the number of images for each set
train_count = 500
test_count = 30
val_count = 30

# Split the images into train, test, and validation sets
train_images = all_images[:train_count]
test_images = all_images[train_count:train_count + test_count]
val_images = all_images[train_count + test_count:]

# Move the images to the corresponding folders
for image in train_images:
    shutil.move(os.path.join(source_folder, image), os.path.join(train_folder, image))

for image in test_images:
    shutil.move(os.path.join(source_folder, image), os.path.join(test_folder, image))

for image in val_images:
    shutil.move(os.path.join(source_folder, image), os.path.join(val_folder, image))

print("Images have been successfully split into train, test, and validation sets.")
