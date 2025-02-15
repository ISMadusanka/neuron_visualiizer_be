# Image Classification API

This project provides a FastAPI-based REST API for image classification. The API accepts image files and returns model predictions along with weights and activation values.

## Features

- Image classification using deep learning
- Model weight and activation visualization
- RESTful API endpoints
- Easy-to-use interface

## Prerequisites

Before running this project, make sure you have Python 3.8+ installed on your system.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ISMadusanka/neuron_visualiizer_be.git
cd neuron_visualiizer_be
```

### 2. Install Dependencies

First, install pipenv if you haven't already:

```bash
# For Windows
pip install pipenv

# For macOS
brew install pipenv

# For Ubuntu/Debian
sudo apt install pipenv
```

Then, install project dependencies:

```bash
pipenv install
```

This will create a virtual environment and install all required packages from the Pipfile.

## Running the Application

1. Activate the virtual environment:
```bash
pipenv shell
```

2. Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Usage

### Test Endpoint

To verify the API is working:
1. Open your browser
2. Navigate to `http://localhost:8000/hello`
3. You should see a welcome message

### Make Predictions

To get predictions for an image:

1. Send a POST request to `http://localhost:8000/predict`
2. Use form-data with key `file` and upload your image
3. The API will return:
   - Model prediction
   - Model weights
   - Activation values

Example using curl:
```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/predict
```

Example input image:

![Example Image](https://www.researchgate.net/profile/Creto-Vidal/publication/321174607/figure/fig3/AS:806993333850113@1569413612260/Example-of-a-MNIST-input-An-image-is-passed-to-the-network-as-a-matrix-of-28-by-28.png)



## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License
