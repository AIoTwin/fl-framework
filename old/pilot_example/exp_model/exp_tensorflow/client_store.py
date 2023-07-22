import flwr as fl
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import GlobalMaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential
from tensorflow import keras

(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.cifar10.load_data()

print("Training Images Shape (x train shape) :", train_images.shape)
print("Label of training images (y train shape) :", train_labels.shape)
print("Test Images Shape (x test shape) :", test_images.shape)
print("Label of test images (y test shape) :", test_labels.shape)

# normalisation
train_images, test_images = train_images / 255, test_images / 255

# cnn
IMG_SHAPE = (32, 32, 3)
# Pre-trained model with MobileNetV2
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
# Freeze the pre-trained model weights
base_model.trainable = (
    True  # allows the weights of the base model to be fine-tuned during training.
)

for layer in base_model.layers[
    :100
]:  # This loop freezes the weights of the first 100 layers in the base_model by setting their trainable attribute to False. This approach is often used to selectively freeze some layers while allowing others to be fine-tuned.
    layer.trainable = False

# Trainable classification head
maxpool_layer = GlobalMaxPooling2D()
prediction_layer = Dense(units=10, activation="softmax")
# Layer classification head with feature detector
model = Sequential([base_model, maxpool_layer, prediction_layer])

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
        # remove ste
        model.fit(
            train_images, train_labels, epochs=2, batch_size=32, steps_per_epoch=521
        )
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
