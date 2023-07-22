import flwr as fl
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.cifar10.load_data()

print("Training Images Shape (x train shape) :", train_images.shape)
print("Label of training images (y train shape) :", train_labels.shape)
print("Test Images Shape (x test shape) :", test_images.shape)
print("Label of test images (y test shape) :", test_labels.shape)

train_images, test_images = train_images / 255, test_images / 255

IMG_SHAPE = (32, 32, 3)
# Pre-trained model with MobileNetV2
model = mobilenet_v2(
    input_shape=IMG_SHAPE, include_top=False, weights=MobileNet_V2_Weights.IMAGENET1K_V2
)
# Freeze the pre-trained model weights
# base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

# Trainable classification head
maxpool_layer = nn.MaxPool2d()

# Layer classification head with feature detector
model = nn.Sequential(base_model, maxpool_layer, prediction_layer)
num_epochs = 10
fine_tune_epochs = 30
total_epochs = num_epochs + fine_tune_epochs
#
# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="Adam", metrics=["sparse_categorical_accuracy"])


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(
            train_images, train_labels, epochs=1, batch_size=32, steps_per_epoch=100
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
fl.client.start_numpy_client(server_address="127.0.0.1:5002", client=CifarClient())
