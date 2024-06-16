import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data

# Preprocess data
preprocess_data('dataset/')

num_classes = 2
input_shape = (224, 224, 3)

# Load MobileNetV2 model with ImageNet weights
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

# Build custom model on top of base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(tf.keras.optimizers.Adam(
    learning_rate=0.0005), loss='binary_crossentropy',
    metrics=['accuracy'])

# Configure data generators for loading and augmenting data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Initialize History callback
history = History()

# Train the model
model.fit(
    train_generator,
    epochs=9,
    validation_data=validation_generator,
    callbacks=[history]
)

# Save the model
model.save('mobilenetv2_food_classifier.h5')

# Plot and save training & validation accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('accuracy_plot.png')

# Plot and save training & validation loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.savefig('loss_plot.png')
