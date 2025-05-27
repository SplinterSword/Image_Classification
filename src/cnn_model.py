import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt

def create_cnn_model(
    input_shape=(32, 32, 3),
    num_classes=10,
    conv_layers=2,
    dense_layers=1,  
    initial_filters=32,
    l2_lambda=0.001,
    filter_growth_rate=2,
    learning_rate=0.001
):
    model = models.Sequential([
        layers.Rescaling(1./255)
    ])
    
    model.add(layers.Conv2D(initial_filters, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(l2_lambda)))
    
    current_filters = initial_filters
    for _ in range(conv_layers):
        current_filters *= filter_growth_rate
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(current_filters, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    
    model.add(layers.Flatten())
    
    for _ in range(dense_layers):
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

def train_model(
    model,
    x_train, 
    y_train, 
    epochs=100,
    batch_size=32,
    early_stopping=True,
    learning_rate_decay=True
):

    callbacks = []
    if early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        callbacks.append(early_stop)
    
    if learning_rate_decay:
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5
        )
        callbacks.append(lr_schedule)
    
    history = model.fit(
        x_train, y_train, 
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history