import tensorflow as tf
from keras import layers, models
from data import load_data
from keras.api.models import load_model

import numpy as np
import matplotlib.pyplot as plt



def get_cnn_model():
    # Build a simple neural network model
    model = models.Sequential([
        layers.Dense(36, activation='relu', input_shape=(36,)),
        layers.Dense(18, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, x_test_resized, y_test):
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test_resized, y_test))
    model.save("minis_v1.keras")
    return history


def load_ann_model():
    return load_model("minis_v1.keras")

if __name__ == "__main__":
    (x_train_resized, y_train), (x_test_resized, y_test) = load_data()
    m = get_cnn_model()
    h = train_model(m, x_train_resized, y_train, x_test_resized, y_test)

