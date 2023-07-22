import time

from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow import keras

IMG_SHAPE = (32, 32, 3)
# Pre-trained model with MobileNetV2
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights=None)
# Freeze the pre-trained model weights
base_model.trainable = True

base_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="Adam",
    metrics=["sparse_categorical_accuracy"],
)


def append_to_file(file_path, content):
    try:
        with open(file_path, "a") as file:
            file.write(content)
            file.write("\n")
    except IOError:
        print("An error occurred while writing to the file.")


def main():
    start_time = time.time()
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = keras.datasets.cifar10.load_data()

    print("Training Images Shape (x train shape) :", train_images.shape)
    print("Label of training images (y train shape) :", train_labels.shape)
    print("Test Images Shape (x test shape) :", test_images.shape)
    print("Label of test images (y test shape) :", test_labels.shape)

    train_images, test_images = train_images / 255, test_images / 255

    batch_size = 2
    learning_rate = 0.001
    num_epochs = 100
    nr_of_samples = 200

    base_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="Adam",
        metrics=["sparse_categorical_accuracy"],
    )

    base_model.fit(
        train_images,
        train_labels,
        epochs=num_epochs,
        batch_size=batch_size,
        steps_per_epoch=(nr_of_samples / batch_size),
    )
    append_to_file("results.txt", "Mobile Net V2 - not pretrained")
    append_to_file(
        "results.txt", "Training time: " + str(round((time.time() - start_time), 2))
    )

    loss, accuracy = base_model.evaluate(test_images, test_labels)
    append_to_file(
        "results.txt", f"Accuracy on test images: {round((accuracy * 100), 2)}%"
    )
    append_to_file("results.txt", "")

    print(loss)
    # Test the model


if __name__ == "__main__":
    main()
