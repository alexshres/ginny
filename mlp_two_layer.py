# Two layer MLP algorithm
import numpy as np
import matplotlib.pyplot as plt

class TwoMLP:
    def __init__(self,
                 train_data,
                 train_labels,
                 hidden_nodes,
                 outputs,
                 epochs,
                 learning_rate,
                 momentum,
                 type,
                 type_value):

        self.train_data = train_data
        self.train_labels = train_labels

        self.hidden_nodes = hidden_nodes
        self.outputs = outputs
        self.epochs = epochs
        self.lr = learning_rate
        self.m = momentum



        # Weights for first and second layer randomly initialized
        self.W = np.random.uniform(-0.5, 0.5, (self.train_data.shape[1], hidden_nodes))
        self.W_prev = np.zeros_like(self.W)

        self.V = np.random.uniform(-0.5, 0.5, (hidden_nodes, outputs))
        self.V_prev = np.zeros_like(self.V)

        # Bias weights randomization
        self.b_W = np.random.uniform(-0.5, 0.5, (1, hidden_nodes))
        self.b_W_prev = np.zeros_like(self.b_W)
        self.b_V = np.random.uniform(-0.5, 0.5, (1, outputs))
        self.b_V_prev = np.zeros_like(self.b_V)

        # Where each input's layer activations will be stored
        self.hidden_activation = None
        self.output_activation = None

        # Will store each layer's error as we back propagate
        self.hidden_error = None
        self.output_error = None

        # Where we will store accuracies
        self.training_accuracy = [0 for i in range(self.epochs)]
        self.test_accuracy = [0 for i in range(self.epochs)]

        # Zero initialized confusion matrix where dimensions = output x output
        self.confusion_matrix = np.zeros((outputs, outputs))

        self.type = type
        self.type_value = type_value


    def __sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def __sigmoid_prime(self, x):
        return x * (1-x)

    def __forward(self, data):

        # Calculating hidden layer activations then pass that into output activations
        self.hidden_activation = self.__sigmoid(np.dot(data, self.W) + self.b_W)
        self.output_activation = self.__sigmoid(np.dot(self.hidden_activation,
                                                       self.V) + self.b_V)

        return self.output_activation


    def __backward(self, data, label):
        # Computing error and delta for the output nodes
        self.output_error = label - self.output_activation
        self.output_delta = self.output_error * self.__sigmoid_prime(self.output_activation)

        # Computing error and delta for the hidden layer nodes
        self.hidden_error = np.dot(self.output_delta, self.V.T)
        self.hidden_delta = self.hidden_error * self.__sigmoid_prime(self.hidden_activation)

        # The update values for each weight matrix
        d_W = np.dot(data.T, self.hidden_delta)
        d_V = np.dot(self.hidden_activation.T, self.output_delta)
        d_b_W = np.sum(self.hidden_delta, axis=0, keepdims=True)
        d_b_V = np.sum(self.output_delta, axis=0, keepdims=True)


        # Updating weights
        self.W += self.lr * d_W + self.m * self.W_prev
        self.V += self.lr * d_V + self.m * self.V_prev
        self.b_W += self.lr * d_b_W + self.m * self.b_W_prev
        self.b_V += self.lr * d_b_V + self.m * self.b_V_prev

        # Updating previous weights for momentum update to weights
        self.W_prev = d_W
        self.V_prev = d_V
        self.b_W_prev = d_b_W
        self.b_V_prev = d_b_V


    def train(self, test_data, test_labels):
        for epoch in range(self.epochs):
            correct = 0
            for i in range(self.train_data.shape[0]):
                data = self.train_data[i].reshape(1, -1)

                # encoding labels so 0.9 for correct classification and 0.1 for rest
                label = np.full((10,), 0.1)
                label[self.train_labels[i]] = 0.9

                outputs = self.__forward(data)
                prediction = np.argmax(outputs)

                if prediction == self.train_labels[i]:
                    correct += 1

                self.__backward(data, label)

            # Calculating accuracy
            self.training_accuracy[epoch] = correct / self.train_data.shape[0]
            self.test_accuracy[epoch] = self.__fit(test_data, test_labels)/test_data.shape[0]

        # Plot of accuracies on one chart
        self.accuracy_plot(self.training_accuracy, self.test_accuracy)

    def __fit(self, test_data, test_labels):
        """
        This function is used to calculate accuracies during training for the test data set
        """
        correct = 0

        for i in range(test_data.shape[0]):
            data = test_data[i].reshape(1, -1)
            outputs = self.__forward(data)
            prediction = np.argmax(outputs)

            if prediction == test_labels[i]:
                correct += 1

        return correct

    def fit(self, test_data, test_labels):
        """
        This function is used to actually test against the test data set once training
        is complete

        """
        correct = 0

        total = test_data.shape[0]
        prediction_count = [0 for i in range(10)]
        actual_count = [0 for i in range(10)]

        for i in test_labels:
            actual_count[i] += 1

        for i in range(test_data.shape[0]):
            data = test_data[i].reshape(1, -1)
            outputs = self.__forward(data)
            prediction = np.argmax(outputs)
            prediction_count[prediction] += 1

            if prediction == test_labels[i]:
                correct += 1


            self.confusion_matrix[prediction, test_labels[i]] += 1


        self.plot_confusion_matrix()



    def accuracy_plot(self, train_acc, test_acc):
        plt.figure(figsize=(10, 6))

        plt.plot(train_acc, label='Training Accuracy', marker='o')
        plt.plot(test_acc,  label='Testing Accuracy', marker='x')

        plt.title('Plot of Accuracy for {self.type}={self.type_value}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.savefig(f"accuracy_plot_{self.type}_{self.type_value}.png")


    def plot_confusion_matrix(self):
        confusion_matrix_normalized = self.confusion_matrix.astype('float') / \
                                      self.confusion_matrix.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
        plt.title(f"Confusion Matrix {self.type}={self.type_value}")
        plt.colorbar()

        # Set x and y labels
        tick_marks = np.arange(self.confusion_matrix.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        # Annotate the matrix with counts and normalized values
        thresh = self.confusion_matrix.max() / 2.
        for i, j in np.ndindex(self.confusion_matrix.shape):
            plt.text(j, i,
             f"{self.confusion_matrix[i, j]}\n({confusion_matrix_normalized[i, j]:.2f})",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if self.confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save the plot as a file
        plt.savefig(f"confusion_matrix_{self.type}_{self.type_value}.png")
