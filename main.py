import keras  # For loading the MNIST dataset.
import numpy as np

# MLP Hyperparameters.
LAYER_SIZES = [784, 16, 16, 10]
LEARNING_RATE = 0.05
NUM_EPOCHS = 10
MINI_BATCH_SIZE = 10


class MLP:
    """
    A simple Multilayer Perceptron (MLP) for classification tasks.

    This MLP consists of fully connected layers with ReLU activation for hidden layers
    and softmax activation for the output layer. The network is trained using stochastic
    gradient descent (SGD) with backpropagation.

    Attributes:
        layer_sizes (list): A list of integers representing the number of neurons in each layer.
        learning_rate (float): The learning rate used for weight updates.
        num_layers (int): The total number of layers in the network.
        weights (list): A list of weight matrices for each connection between layers.
        biases (list): A list of bias vectors for each connection between layers.
    """

    def __init__(self, layer_sizes, learning_rate):
        """
        Initializes the MLP with random weights and biases.

        Args:
            layer_sizes (list): A list of integers representing the number of neurons in each layer. For example,
                                `layer_sizes=[10, 5, 5, 2]` initializes a network with an input layer of 10 neurons,
                                two hidden layers with 5 neurons each, and an output layer with 2 neurons.
            learning_rate (float): The learning rate for weight updates.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * np.sqrt(2.0 / layer_sizes[i - 1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(y, 1) * 0.01 for y in layer_sizes[1:]]

    def forward_propagation(self, input_activations):
        """
        Performs a full forward pass through the network.

        For each layer, this method computes the linear combination of weights and inputs,
        applies an activation function (ReLU for hidden layers, softmax for the output),
        and stores both the pre-activation values and the final activations.

        Args:
            input_activations (numpy.ndarray): Input data of shape (input_size, batch_size).

        Returns:
            tuple:
                - activations (list of numpy.ndarray): Activations for each layer, including input and output.
                - zs (list of numpy.ndarray): Pre-activation values (z = Wx + b) for each layer, excluding input.
        """
        activations = [input_activations]
        zs = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(softmax(z) if i == self.num_layers - 2 else relu(z))

        return activations, zs

    def back_propagation(self, input_activations, y):
        """
        Computes gradients for all weights and biases using backpropagation.

        This method performs a forward pass to get activations and pre-activations,
        then computes the output error using the cross-entropy loss (with softmax), and
        propagates this error backward through the network. It returns the gradients
        for each weight and bias, which can be used for updating parameters during training.

        Args:
            input_activations (numpy.ndarray): Input data of shape (input_size, batch_size).
            y (numpy.ndarray): One-hot encoded labels of shape (num_classes, batch_size).

        Returns:
            tuple:
                - cost_gradients_w (list of numpy.ndarray): Gradients of the cost w.r.t. each weight matrix.
                - cost_gradients_b (list of numpy.ndarray): Gradients of the cost w.r.t. each bias vector.
        """
        # Run forward propagation.
        activations, zs = self.forward_propagation(input_activations)

        # Initialise gradient arrays.
        cost_gradients_b = [np.zeros(b.shape) for b in self.biases]
        cost_gradients_w = [np.zeros(w.shape) for w in self.weights]

        # Output layer error (prediction - target).
        delta = activations[-1] - y
        cost_gradients_b[-1] = np.sum(delta, axis=1, keepdims=True)
        cost_gradients_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate through hidden layers.
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * relu_derivative(zs[-l])
            cost_gradients_b[-l] = np.sum(delta, axis=1, keepdims=True)
            cost_gradients_w[-l] = np.dot(delta, activations[-l - 1].T)

        return cost_gradients_w, cost_gradients_b

    def train(self, x_train, y_train, epochs, batch_size):
        """
        Trains the MLP using mini-batch stochastic gradient descent (SGD).

        For each epoch, the training data is shuffled and divided into batches. For each mini-batch,
        the method computes the gradients using backpropagation and updates the model parameters
        (weights and biases) accordingly.

        Args:
            x_train (numpy.ndarray): Input training data of shape (input_size, num_samples).
            y_train (numpy.ndarray): One-hot encoded training labels of shape (num_classes, num_samples).
            epochs (int): Number of full passes through the training data.
            batch_size (int): Number of samples per mini-batch used in gradient updates.
        """
        # Number of training examples.
        n = x_train.shape[1]

        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            x_shuffled = x_train[:, permutation]
            y_shuffled = y_train[:, permutation]

            for k in range(0, n, batch_size):
                x_batch = x_shuffled[:, k:k + batch_size]
                y_batch = y_shuffled[:, k:k + batch_size]
                current_batch_size = x_batch.shape[1]

                cost_gradients_w, cost_gradients_b = self.back_propagation(x_batch, y_batch)

                # Apply gradient update.
                self.weights = [w - (self.learning_rate / current_batch_size) * dw for w, dw in zip(self.weights, cost_gradients_w)]
                self.biases = [b - (self.learning_rate / current_batch_size) * db for b, db in zip(self.biases, cost_gradients_b)]

            print(f"Epoch {epoch + 1} complete")

    def evaluate(self, x_test, y_test):
        """
        Computes classification accuracy of the MLP on the given test dataset.

        Performs a forward pass on the test inputs and compares predicted class labels
        to the true labels.

        Args:
            x_test (numpy.ndarray): Test input data of shape (input_size, num_samples).
            y_test (numpy.ndarray): One-hot encoded test labels of shape (num_classes, num_samples).

        Returns:
            float: Classification accuracy as a value between 0 and 1.
        """
        predictions = np.argmax(self.forward_propagation(x_test)[0][-1], axis=0)
        labels = np.argmax(y_test, axis=0)
        return np.mean(predictions == labels)

    def predict(self, x):
        """
        Returns predicted class labels for the given input data.

        Args:
            x (numpy.ndarray): Input data of shape (input_size, num_samples).

        Returns:
            numpy.ndarray: A 1D array of integers representing predicted class indices,
                           one for each input sample.
        """
        return np.argmax(self.forward_propagation(x)[0][-1], axis=0)


#### Miscellaneous functions
def relu(z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function to the input array.
    ReLU returns 0 for inputs ≤ 0, otherwise it returns the input.

    Args:
        z (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Input array with ReLU applied to each element.
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Applies the derivative of the ReLU activation function to the input array.

    ReLU derivative is 1 for inputs > 0, otherwise it is 0.

    Args:
        z (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Input array with ReLU derivative applied to each element.
    """
    return (z > 0).astype(float)


def softmax(z):
    """
    Applies the softmax activation function to the input array. Softmax takes an array of integers
    as input and transforms them into a probability distribution.

    Args:
        z (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Input array with softmax applied to each element.
    """
    # Fix for numerical stability (prevents overflow without changing the functions output).
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def one_hot_encoded(y):
    """
    Converts a 1D array of integer class labels into one-hot encoded format.

    For each integer label in the input array, this function generates a column vector of zeros
    with a 1 placed at the index corresponding to the label value.

    Args:
        y (numpy.ndarray): A 1D array of integer class labels.

    Returns:
        numpy.ndarray: A 2D array where each row/column represents a one-hot encoded integer label.
    """
    num_classes = np.max(y) + 1
    one_hot_y = np.zeros((num_classes, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y


if __name__ == "__main__":
    # Load the MNIST database, which contains labelled 28x28 pixel grayscale images of handwritten digits (0–9).
    # Each pixel value ranges from 0 (inactive) to 255 (active).
    (training_images, training_labels), (testing_images, testing_labels) = keras.datasets.mnist.load_data()

    # Training data.
    training_images = training_images.reshape(60000, 784).T / 255.0
    training_labels = one_hot_encoded(training_labels)

    # Testing data.
    testing_images = testing_images.reshape(10000, 784).T / 255.0
    testing_labels = one_hot_encoded(testing_labels)

    # Initialise, train and evaluate a new MLP using the given hyperparameters.
    model = MLP(LAYER_SIZES, learning_rate=LEARNING_RATE)
    model.train(training_images, training_labels, epochs=NUM_EPOCHS, batch_size=MINI_BATCH_SIZE)
    accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
