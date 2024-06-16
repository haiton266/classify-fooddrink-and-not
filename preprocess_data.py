import os
import shutil
import random


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def preprocess_data(dataset_path):
    valid_path = os.path.join(dataset_path, 'valid')
    notvalid_path = os.path.join(dataset_path, 'notvalid')
    train_valid_path = 'data/train/valid'
    train_notvalid_path = 'data/train/notvalid'
    val_valid_path = 'data/val/valid'
    val_notvalid_path = 'data/val/notvalid'

    # Clear old data
    clear_directory(train_valid_path)
    clear_directory(train_notvalid_path)
    clear_directory(val_valid_path)
    clear_directory(val_notvalid_path)

    # Get all images in valid and notvalid folders
    valid_images = [os.path.join(valid_path, img)
                    for img in os.listdir(valid_path)]
    notvalid_images = [os.path.join(notvalid_path, img)
                       for img in os.listdir(notvalid_path)]

    # Shuffle and split the data
    random.shuffle(valid_images)
    random.shuffle(notvalid_images)

    split_index_valid = int(0.7 * len(valid_images))
    split_index_notvalid = int(0.7 * len(notvalid_images))

    train_valid_images = valid_images[:split_index_valid]
    val_valid_images = valid_images[split_index_valid:]

    train_notvalid_images = notvalid_images[:split_index_notvalid]
    val_notvalid_images = notvalid_images[split_index_notvalid:]

    # Copy images to train and val folders
    for img in train_valid_images:
        shutil.copy(img, train_valid_path)

    for img in val_valid_images:
        shutil.copy(img, val_valid_path)

    for img in train_notvalid_images:
        shutil.copy(img, train_notvalid_path)

    for img in val_notvalid_images:
        shutil.copy(img, val_notvalid_path)
