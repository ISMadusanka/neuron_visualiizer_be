from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from data import process_image
from predict import get_predict
import uvicorn

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """ Receive image, preprocess, and send to ANN model """
    contents = await file.read()
    image = Image.open(BytesIO(contents))  # Open image
    processed_image = process_image(image)
    prediction = get_predict(processed_image)

    return {"prediction":str(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# uvicorn app:app --host 0.0.0.0 --port 8000 --reload