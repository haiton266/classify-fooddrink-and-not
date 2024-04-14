from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đọc mô hình đã lưu
saved_model = load_model('mobilenetv2_food_classifier.h5')

# Load và augment dữ liệu kiểm thử
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),  # (299, 299)
    batch_size=32,
    class_mode='binary')  # Đảm bảo chỉ định 'binary' cho binary classification

# Đánh giá mô hình trên tập test
test_loss, test_accuracy = saved_model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
