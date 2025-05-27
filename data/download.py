import tensorflow as tf
import os
import logging
import numpy as np
from datetime import datetime

def download_cifar(
    save_path='data/raw',
    dataset_name='cifar10',
    validate_download=True
):
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    
    logger = logging.getLogger(__name__)

    os.makedirs(save_path, exist_ok=True)

    try:
        # Start download timer
        start_time = datetime.now()
        
        # Download dataset
        if dataset_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif dataset_name == 'cifar100':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        else:
            raise ValueError("Unsupported dataset. Choose 'cifar10' or 'cifar100'.")
        

        if validate_download:
            logger.info(f"Validating {dataset_name} download...")

            assert len(x_train) > 0, "Training data is empty"
            assert len(x_test) > 0, "Test data is empty"

            logger.info(f"Training images: {len(x_train)}")
            logger.info(f"Test images: {len(x_test)}")


        # Calculate download duration
        download_duration = datetime.now() - start_time
        logger.info(f"Download completed in {download_duration}")


        return (x_train, y_train), (x_test, y_test)


    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e


def save_dataset(data, save_path='data/raw'):
    (x_train, y_train), (x_test, y_test) = data

    os.makedirs(save_path, exist_ok=True)
    
    np.save(os.path.join(save_path, 'x_train.npy'), x_train)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'x_test.npy'), x_test)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)

if __name__ == '__main__':
    dataset = download_cifar()
    save_dataset(dataset)