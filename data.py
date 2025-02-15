
import tensorflow as tf
import numpy as np
from PIL import Image

def load_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Resize images to 6x6
    x_train_resized = tf.image.resize(x_train[..., np.newaxis], (6, 6)).numpy().reshape(-1, 36)
    x_test_resized = tf.image.resize(x_test[..., np.newaxis], (6, 6)).numpy().reshape(-1, 36)

    # Normalize pixel values
    x_train_resized, x_test_resized = x_train_resized / 255.0, x_test_resized / 255.0

    return (x_train_resized, y_train), (x_test_resized, y_test)


def process_image(image: Image.Image):
    """Resize and normalize an input image for the ANN model"""
    # Convert PIL image to NumPy array (Grayscale)
    image_array = np.array(image.convert("L"))  # Convert to grayscale

    # Add batch dimension and channel axis for TensorFlow compatibility
    image_array = image_array.astype(np.float32) / 255.0  # Normalize (0-1)

    # Resize image to 6x6 using TensorFlow
    image_resized = tf.image.resize(image_array[..., np.newaxis], (6, 6)).numpy().reshape(-1, 36)

    return image_resized