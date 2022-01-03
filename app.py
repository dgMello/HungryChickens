import boto3
import os
import PIL.Image as Image
import boto3
import time
import tensorflow as tf
import numpy as np
import logging

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#from secrets import access_key, secret_access_key
import pathlib
from notifier import msgSender

logger = logging.getLogger(__name__)

img_width = 180
img_height = 180
img_shape = (img_width, img_height)
class_names = ['american goldfinch', 'black-capped chickadee', 'house sparrow', 'northern cardinal', 'northern mockingbird']
model = tf.keras.models.load_model("blink_birder_model")
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    logger.info(bucket_name)
    key = event['Records'][0]['s3']['object']['key']
    logger.info(key)
    img = readImageFromBucket(key, bucket_name).resize(img_shape)
    img = tf.keras.utils.img_to_array(img, data_format=None, dtype=None)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    logger.info(predictions)
    score = tf.nn.softmax(predictions[0])
    logger.info(score)
    scaledScore = 100 * np.max(score)
    if scaledScore >= 90:
        bird = class_names[np.argmax(score)]
        msgSender(bird, True)
    else:
        bird = "possibly_bird"
        msgSender(bird, False)


def readImageFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name)
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])