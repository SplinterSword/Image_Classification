import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

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
    normalize=True
):

    processed_x_train = x_train
    processed_y_train = y_train

    if normalize:
        processed_x_train = normalize_data(processed_x_train)
    
    return processed_x_train, processed_y_train

# def split_training_set(x_train, y_train, validation_split=0.2, random_state=42):

#     x_train, x_val, y_train, y_val = train_test_split(
#         x_train, 
#         y_train, 
#         test_size=validation_split, 
#         stratify=y_train, 
#         random_state=random_state
#     )

#     return x_train, x_val, y_train, y_val

def save_dataset(processed_x_train, processed_y_train, save_path='data/processed'):
    
    # x_train, x_val, y_train, y_val = split_training_set(processed_x_train, processed_y_train)

    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, 'x_train.npy'), x_train)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)

if __name__ == '__main__':
    x_train = np.load("/home/splintersword/Documents/Projects/Image_Classification/data/raw/x_train.npy")
    y_train = np.load("/home/splintersword/Documents/Projects/Image_Classification/data/raw/y_train.npy")

    processed_x_train, processed_y_train = preprocess_cifar_data(x_train, y_train, False)

    save_dataset(processed_x_train, processed_y_train)