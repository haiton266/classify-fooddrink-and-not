from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from starlette.concurrency import run_in_threadpool

app = FastAPI()

# Asynchronous function to load the model


async def get_model():
    model = await run_in_threadpool(load_model, 'mobilenetv2_food_classifier.h5')
    return model

model = None  # Initialize model variable


@app.on_event("startup")
async def load_model_on_startup():
    global model
    model = await get_model()  # Load the model asynchronously


def predict_image_from_stream(img_stream):
    try:
        img = image.load_img(img_stream, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # Normalize the image data
        prediction = model.predict(x)
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - \
            prediction[0][0]
        return True if prediction[0][0] > 0.5 else False
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        image_data = await file.read()
        img_stream = BytesIO(image_data)
        prediction = predict_image_from_stream(img_stream)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
