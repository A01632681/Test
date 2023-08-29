# Description: This script is used to train a CNN for image classification using TensorFlow and Keras.
import tensorflow as tf
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Disable eager execution to avoid errors
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Define the paths to your dataset
train_data_dir = r"D:\Documentos\Dissertation\Images\seg_train"
val_data_dir = r'D:\Documentos\Dissertation\Images\seg_pred'
test_data_dir = r'D:\Documentos\Dissertation\Images\seg_test'

# Set the input image dimensions
input_shape = (224, 224, 3)
num_classes = 6

# Define the batch size and number of epochs
batch_size = 32
num_epochs = 10

# Create data generators for preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# # Define a function for image preprocessing
# def preprocess_images(image_dir, target_size):
#     datagen = ImageDataGenerator(rescale=1./255)
#     generator = datagen.flow_from_directory(
#         image_dir,
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=True
#     )
#     return generator

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

# Define the detection network architecture
def detection_network():
    model = tf.keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Create an instance of the detection network
model = detection_network()

# Define the callbacks for early stopping and saving the best model
callbacks = [EarlyStopping(monitor='accuracy', patience=1),
             ModelCheckpoint("model.keras", save_best_only=True, monitor="accuracy", mode='max')]

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
# Train the model
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=num_epochs,
          #validation_data=val_generator,
          #validation_steps=len(val_generator),
          callbacks=callbacks)

# Load the best model
model = models.load_model("model.keras")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', test_acc)

models.save_model(model, "Model.h5",save_format='h5')