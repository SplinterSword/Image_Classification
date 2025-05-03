import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

def data_augmentation(x_train, y_train):

    data_augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1)
    ])
    

    augmented_x_train = data_augmentation_layer(x_train)
    
    augmented_y_train = np.repeat(y_train, 2, axis=0)
    
    return augmented_x_train, augmented_y_train

def normalize_data(x_train):

    x_min = x_train.min()
    x_max = x_train.max()
    
    if x_max == x_min:
        return x_train
    
    normalized_x = (x_train - x_min) / (x_max - x_min)
    
    return normalized_x


def preprocess_cifar_data(
    x_train,
    y_train,
    augment=True, 
    normalize=True
):

    processed_x_train = x_train
    processed_y_train = y_train
    
    if augment:
        augmented_x_train, augmented_y_train = data_augmentation(processed_x_train, processed_y_train)
        processed_x_train = np.concatenate([processed_x_train, augmented_x_train], axis=0)
        processed_y_train = augmented_y_train

    if normalize:
        processed_x_train = normalize_data(processed_x_train)
    
    return processed_x_train, processed_y_train

def split_training_set(x_train, y_train, validation_split=0.2, random_state=42):

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, 
        y_train, 
        test_size=validation_split, 
        stratify=y_train, 
        random_state=random_state
    )

    return x_train, x_val, y_train, y_val

def save_dataset(processed_x_train, processed_y_train, save_path='data/processed'):
    
    x_train, x_val, y_train, y_val = split_training_set(processed_x_train, processed_y_train)

    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, 'x_train.npy'), x_train)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'x_val.npy'), x_val)
    np.save(os.path.join(save_path, 'y_val.npy'), y_val)


if __name__ == '__main__':
    x_train = np.load("/home/splintersword/Documents/Computers/Image_Classification/data/raw/x_train.npy")
    y_train = np.load("/home/splintersword/Documents/Computers/Image_Classification/data/raw/y_train.npy")

    processed_x_train, processed_y_train = preprocess_cifar_data(x_train, y_train)

    save_dataset(processed_x_train, processed_y_train)

