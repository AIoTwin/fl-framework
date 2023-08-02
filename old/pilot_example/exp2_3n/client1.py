from tensorflow import keras
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from matplotlib import pyplot
from keras import datasets

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import flwr as fl

(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.cifar10.load_data()

classes_to_keep_train = [0, 1, 2]
train_indices = [
    i for i in range(len(train_labels)) if train_labels[i][0] in classes_to_keep_train
]

classes_to_keep_test = [0, 1, 2, 3, 4, 5, 6, 7, 8]
test_indices = [
    i for i in range(len(test_labels)) if test_labels[i][0] in classes_to_keep_test
]

train_images = train_images[train_indices]
train_labels = train_labels[train_indices]

test_images = test_images[test_indices]
test_labels = test_labels[test_indices]


print("Training Images Shape (x train shape) :", train_images.shape)
print("Label of training images (y train shape) :", train_labels.shape)
print("Test Images Shape (x test shape) :", test_images.shape)
print("Label of test images (y test shape) :", test_labels.shape)

train_images, test_images = train_images / 255, test_images / 255

from keras.applications.mobilenet_v2 import MobileNetV2

IMG_SHAPE = (32, 32, 3)
# Pre-trained model with MobileNetV2
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
# Freeze the pre-trained model weights
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False


# Trainable classification head
maxpool_layer = GlobalMaxPooling2D()
prediction_layer = Dense(units=10, activation="softmax")
# Layer classification head with feature detector
model = Sequential([base_model, maxpool_layer, prediction_layer])
num_epochs = 10
fine_tune_epochs = 30
total_epochs = num_epochs + fine_tune_epochs

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="Adam",
    metrics=["sparse_categorical_accuracy"],
)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_images, train_labels, epochs=1, batch_size=32)
        return model.get_weights(), len(train_images), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels)

        f = open("eval.csv", "a")
        f.write(str(loss) + "," + str(accuracy) + "\n")
        f.close()

        return loss, len(test_images), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:5050", client=CifarClient())
