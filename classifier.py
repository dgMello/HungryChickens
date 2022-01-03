import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import boto3
import time
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#from secrets import access_key, secret_access_key
import pathlib


class CreateModel:
    def __init__(self, dataset_path, img_height, img_width, batch_size=32):
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_ds = None
        self.val_ds = None
        self.class_names = None
        self.model = None
        self.history = None

    def create_training_validation_data(self):
        data_dir = pathlib.Path(self.dataset_path)

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names

    def create_classifier_model(self):
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.Rescaling(1. / 255)

        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))

        num_classes = len(self.class_names)

        self.model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

    def train_model(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

        epochs = 10
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )
        self.model.save('blink_birder_model')

    def graph_history(self, epochs=10):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def test_model(self, test_image_path):
        test_image_path = test_image_path

        img = tf.keras.utils.load_img(
            test_image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )


# bird_model = CreateModel("C:\\Users\\dougmell\\PycharmProjects\\BlinkQAPythonFramework\\TheHungryChickens\\datasets\\birds", 180, 180)
# bird_model.create_training_validation_data()
# bird_model.create_classifier_model()
# bird_model.train_model()
# bird_model.graph_history()
# bird_model.test_model("C:\\Users\\dougmell\\Pictures\\bird_test.jpg")


def detect_bird_type(model_path, image):
    class_names = ['american goldfinch', 'black-capped chickadee', 'house sparrow', 'northern cardinal', 'northern mockingbird']
    new_model = tf.keras.models.load_model(model_path)
    test_image_path = image

    img = tf.keras.utils.load_img(
        test_image_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    #print(
    #    "This image most likely belongs to {} with a {:.2f} percent confidence."
    #        .format(class_names[np.argmax(score)], 100 * np.max(score))
    #)
    scaledScore = 100 * np.max(score)

    if scaledScore >= 90:
        return class_names[np.argmax(score)]
    else:
        return "possibly_bird"

def get_image_files():
    s3 = boto3.client('s3', aws_access_key_id='',
                      aws_secret_access_key='')
    s3.download_file('blinkbird', 'Bird_northern_mockingbird.jpeg', 'Bird_northern_mockingbird.jpeg')
    time.sleep(30)
    image_path = "C:\\Users\dougmell\\PycharmProjects\\BlinkQAPythonFramework\\TheHungryChickens\\Bird_northern_mockingbird.jpeg"
    return image_path


image = get_image_files()
PIL.Image.open(image)
detect_bird_type('blink_birder_model', image)
