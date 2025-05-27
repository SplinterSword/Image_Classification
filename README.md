# Image Classification using CNN on CIFAR-10

A deep learning project that implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. This project demonstrates the implementation of a robust image classification pipeline using TensorFlow and Keras.

## Project Structure

```
.
├── data/                    # Dataset and data processing scripts
│   ├── README.md           # Dataset documentation
│   └── download.py         # Script to download the CIFAR-10 dataset
├── src/                    # Source code
│   ├── cnn_model.ipynb     # Main Jupyter notebook with model implementation
│   ├── models/             # Directory to store trained models
│   └── *.jpg               # Example images for testing
├── tests/                  # Test files
│   └── test_download.py    # Tests for data download functionality
├── Pipfile                 # Python dependencies
└── Pipfile.lock            # Locked dependencies
```

## Features

- **Data Augmentation**: Implements various augmentation techniques including random flips, rotations, translations, and zoom
- **CNN Architecture**: Custom CNN model with multiple convolutional and dense layers
- **Training Pipeline**: Complete training loop with learning rate scheduling
- **Model Evaluation**: Includes accuracy, loss visualization, and confusion matrix
- **Model Persistence**: Save and load trained models for inference

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook (for development)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Image_Classification
   ```

2. Install dependencies using Pipenv:
   ```bash
   pip install pipenv
   pipenv install
   pipenv shell
   ```

3. Download the CIFAR-10 dataset:
   ```bash
   python data/download.py
   ```

## Usage

1. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook src/cnn_model.ipynb
   ```

2. The notebook is organized into the following sections:
   - Environment setup and imports
   - Data loading and preprocessing
   - Model architecture definition
   - Model training with data augmentation
   - Model evaluation and visualization
   - Saving and loading the trained model

## Model Architecture

The CNN model consists of:
- Multiple convolutional blocks with batch normalization
- Max-pooling layers for downsampling
- Dropout layers for regularization
- Dense layers for classification
- Learning rate scheduling for better convergence

## Results

The model achieves competitive accuracy on the CIFAR-10 test set. Detailed performance metrics and visualizations are available in the notebook.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
