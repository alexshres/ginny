import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, inputs, outputs=1, learning_rate=0.01, epochs=100):
        self.inputs = inputs
        self.outputs = outputs

        # Hyperparamters
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initializing our weights to be all zeroes
        # Shape of weights matrix is number of outputs times number of paramters in X
        self.weights = np.random.uniform(-0.05, 0.05, (self.outputs, self.inputs)) 

        self.confusion_matrix = np.zeros((outputs, outputs))
        self.epoch_accuracy = [0 for i in range(epochs)]

        if self.learning_rate == 0.001:
            self.strng = "0_001"
        elif self.learning_rate == 0.01:
            self.strng = "0_01"
        else:
            self.strng = "0_1"



    def train(self, X, y):

        ones_column = np.ones((X.shape[0], 1))
        training_data = np.hstack((ones_column, X))

        labels = y
        encoded_y = np.zeros((y.size, y.max() + 1)) 
        encoded_y[np.arange(y.size), y] = 1

        for e in range(self.epochs):
            # for each training data 
            target_counter = [ 0 for i in range(10)]
            predicted_counter = [ 0 for i in range(10)]

            for i in range(training_data.shape[0]):

                # Getting predicted class label 
                output_vector = self.weights @ training_data[i].T 
                prediction = np.argmax(output_vector)

                target_counter[labels[i]] += 1
                if prediction == labels[i]:
                    predicted_counter[prediction] += 1


                for j in range(self.weights.shape[0]):
                    # Updating how far off we were
                    t = encoded_y[i, j]
                    if output_vector[j] > 0 and j == prediction:
                        y = 1 
                    else:
                        y = 0
                    self.weights[j] = self.weights[j] + self.learning_rate * \
                            (t-y) * training_data[i]

            # Updating accuracy at each epoch
            self.epoch_accuracy[e] = np.sum(predicted_counter)/np.sum(target_counter)

        self.accuracy_plot() 

        return 

    def predict(self, X, y):
        test_data = X
        ones_column = np.ones((X.shape[0], 1))
        test_data = np.hstack((ones_column, test_data))

        test_labels = y
        total = test_data.shape[0]
        correct = 0
        prediction_count = [0 for i in range(10)]
        actual_count = [0 for i in range(10)]

        for i in test_labels:
            actual_count[i] += 1

        for i in range(test_data.shape[0]):
            output = self.weights @ test_data[i].T
            prediction = np.argmax(output)
            prediction_count[prediction] += 1

            if prediction == test_labels[i]:
                correct += 1

            self.confusion_matrix[prediction, test_labels[i]] += 1


        print(f"Model_{self.strng} predicted {correct} out of {total} for a {(1.0 * correct)/total} accuracy")

        self.plot_confusion_matrix()



    def accuracy_plot(self):
        labels = list(range(len(self.epoch_accuracy)))
        plt.figure(figsize=(10, 5))
        plt.plot(labels, self.epoch_accuracy, marker='o')

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Plot of Accuracy by Epoch eta = {self.strng}")

        plt.savefig(f"accuracy_plot_{self.strng}.png")



    def ols(self) -> np.ndarray:  # returns best fit line
        x = np.zeros(self.X.shape[0])
        A_t = np.linalg.matrix_transpose(self.X)
        inv_sym = np.linalg.inv(A_t @ self.X)         # matmul
        half_proj = inv_sym @ A_t
        x = half_proj @ b

        return x

    def plot_confusion_matrix(self):
        confusion_matrix_normalized = self.confusion_matrix.astype('float') / \
                                      self.confusion_matrix.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap='Blues')
        plt.title(f"Confusion Matrix eta = {self.strng}")
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
        plt.savefig(f"confusion_matrix_{self.strng}.png")
