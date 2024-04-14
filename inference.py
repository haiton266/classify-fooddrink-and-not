import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # (299, 299)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Chuẩn hóa dữ liệu
    prediction = model.predict(x)
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - \
        prediction[0][0]
    if prediction[0][0] < 0.5:
        return {'result': "Không phải đồ ăn thức uống", 'confidence': confidence}
    else:
        return {'result': "Đồ ăn thức uống", 'confidence': confidence}


loaded_model = load_model('mobilenetv2_food_classifier.h5')
image_path = 'dataset/test/notvalid/5.jpg'
prediction = predict_image(image_path, loaded_model)
print(prediction)
