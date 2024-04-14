import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cấu hình cơ bản
num_classes = 2
input_shape = (224, 224, 3)

# Load mô hình MobileNetV2 với trọng số ImageNet
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

# Xây dựng mô hình tùy chỉnh phía trên mô hình cơ sở
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Cấu hình data generators để load và augment dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Huấn luyện mô hình
model.fit(
    train_generator,
    epochs=9,
    validation_data=validation_generator,
)

# Lưu mô hình
model.save('mobilenetv2_food_classifier.h5')
