import argparse
import os
from pathlib import Path

import tensorflow as tf


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class Local:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return self.model.get_weights()  

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters

        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

    def validate(self, parameters, config, x_eval, y_eval):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(x_eval, y_eval, 32, steps=steps)
        num_examples_test = len(x_eval)
        return loss, num_examples_test, {" validation accuracy": accuracy}

def main() -> None:
    config=dict()
    config["val_steps"]=10
    config["local_epochs"]=config["epochs"]=4
    config["batch_size"]=32
    # Parse command line argument `partition`
   
    # Load and compile Keras model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test),(x_eval,y_eval) = load_data()
    

    client = Local(model, x_train, y_train, x_test, y_test)
    parameters=client.get_parameters()
    for i in range(1):
        parameters,num_examples_train, results=client.fit(parameters,config)
        loss, num_examples_test, accuracy=client.evaluate(parameters,config)
        loss, num_examples_test, accuracy=client.validate(parameters,config,x_eval,y_eval)


def load_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_eval=x_train[55000:60000]
    y_eval=y_train[55000:60000]
    x_train, x_test,x_eval = x_train / 255.0, x_test / 255.0, x_eval/255.0
    
    return (
        x_train[0:55000],
        y_train[0:55000],
    ), (
        x_test[0:55000],
        y_test[0:55000],
    ), (x_eval, y_eval)


if __name__ == "__main__":
    main()
