from models import load_ann_model
from data import load_data
import numpy as np


model = load_ann_model()
(x_train_resized, y_train), (x_test_resized, y_test) = load_data()

def predict_single_image(index):
    image = x_test_resized[index].reshape(1, -1)
    activations = image
    print("Model Weights:")
    for layer in model.layers:
        weights, biases = layer.get_weights()
        print(f'Layer: {layer.name}\nWeights: {weights}\nBiases: {biases}\n')
        activations = layer(activations)
        print(f'Activation Output: {activations.numpy()}\n')

    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    print(f'Predicted Label: {predicted_label}, True Label: {y_test[index]}')
    # plt.imshow(x_test[index], cmap='gray')
    # plt.show()


import numpy as np
import json


def get_predict(image):
    activations = image
    details = {"model_weights": []}

    for layer in model.layers:
        weights, biases = layer.get_weights()
        layer_info = {
            "layer_name": layer.name,
            "weights": weights.tolist(),  # Convert to list for JSON serialization
            "biases": biases.tolist()  # Convert to list for JSON serialization
        }
        details["model_weights"].append(layer_info)

        activations = layer(activations)
        details["activation_output"] = activations.numpy().tolist()  # Convert to list for JSON serialization

    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    details["predicted_label"] = int(predicted_label)  # Convert to int for JSON serialization

    return json.dumps(details, indent=4)  # Return as structured JSON


if __name__ == "__main__":
    while True:
        n = int(input("Enter an index: "))
        predict_single_image(n)