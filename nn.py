# TODO finish backpropagation algorithm 

import numpy as np


class NeuralNetwork:
    def __init__(self, train_data, train_labels,
                 nodes,
                 epochs=10,
                 learning_rate=0.1,
                 momentum=0.9
                 ):
        """
        Input size will be included in the nodes array where the first index
        contains the number of parameters in training data plus one (for the biases)
        Each subsequent index will hold the number of nodes for each hidden layer:
            nodes[0] = # of parameters
                .
                .
            nodes[i] = # of nodes in i_th hidden layer
                .
                .
            nodes[n] = # of nodes in output
        """

        # Augmenting training data with biases of 1
        self.train_data = train_data
        self.train_labels = train_labels
        self.epochs = epochs
        self.eta = learning_rate
        self.nodes = nodes
        self.layers = len(nodes) - 1
        self.momentum = momentum

        # Each index in the array self.layer_weights corresponds to weights
        # from one layer to its next layer
        self.layer_weights = []
        for layer in range(self.layers):
            self.layer_weights.append(np.random.uniform(-0.05, 0.05,
                                                 (nodes[layer], nodes[layer+1])))

        # Bias weights
        self.bias_weights = []
        for layer in range(self.layers):
            self.bias_weights.append(np.random.uniform(-0.05, 0.05, (1, nodes[layer])))

        # Will be updated by the forward method
        self.__activations = []

        # Will be updated by the backward method
        self.__errors = []

        print(f"Printing initialized weights:\n{self.layer_weights}")


    def __sigmoid(self, x):
        """ Implementation of sigmoid function"""
        return 1/(1+np.exp(-x))

    def __dsigmoid(self, x):
        """ Derivative of sigmoid function"""
        return x * (1-x)


    def __forward(self, data):
        """
        Forward propagation of finding h_j where h_j is the activated value
        of the j_th node
        """

        current_activation = data
        for lr in range(self.layers):
            self.__activations.append(self.__sigmoid(
                                            np.dot(current_activation,
                                                   self.layer_weights[lr]) + self.bias_weights[lr])
                                     )
            current_activation = self.__activations[-1]

        # This will return the activation of the output layer
        return self.__activations[-1]

    def __backprop(self, target, data):
        """
        Backpropagation portion of the algorithm
        """

        error = []
        encoding = [0.9 if x == target else 0.1 for x in range(self.nodes[-1])]
        print(f"Encoding is {encoding}")


        # grabbing just the acivations we need to compute errors for current
        # input row
        self.__activations

        # Reversing for backpropagation to work
        rev_activations = self.__activations[::-1]

        # getting output error
        for t in range(len(encoding)):
            output = rev_activations[0][t]
            target = encoding[t]
            error.append(output * (1-output) * (target-output))

        self.__errors.append(np.asarray(error))

        # Calculating layer error at each hidden node and then
        # adding it to self.__errors so we can access later for updates
        # and previous layer error calculations
        for i in range(1, len(rev_activations)):
            layer_error = []
            activations = rev_activations[i]
            # Removing the 1 that corresponds to the bias
            activations = activations[-len(activations)+1:]
            previous_layer_error = self.__errors[-1]

            for j in range(len(activations)):
                wt_idx = len(self.layer_weights)-i
                error = activations[j] * (1 - activations[j]) * np.dot(previous_layer_error,
                                                                       self.layer_weights[wt_idx][j+1, :])
                layer_error.append(error)
            self.__errors.append(np.asarray(layer_error))

        print(f"Printing self.__errors = {self.__errors}")
        # Now that we have all the errors we can updating weights from input layer all the
        # way to the last layer, that's why we have to reverse the errors
        rev_errors = self.__errors[::-1]


        print(f"self.__activations = {self.__activations}")


        # Updating weights now
        for i in range(len(self.layer_weights)):
            for j in range(self.layer_weights[i].shape[0]):
                for k in range(self.layer_weights[i].shape[1]):
                    err = rev_errors[i][k]

                    print(f"self.__activations[i] = {self.__activations[i]}")
                    print(f"shape is {self.__activations[i].shape}")
                    print(f"j = {j}")
                    active = self.__activations[i][j]

                    self.layer_weights[i][j, k] = self.eta * err * active


        print(f"Printing updated weights {self.layer_weights}")



    def train(self):
        for row in range(self.train_data.shape[0]):
            target = self.train_labels[row]
            data = self.train_data[row]

            print("Beginning forward propagation.")
            self.__forward(data)


            """
            print(f"Printing activations:\n{self.__activations}")


            print("Beginning back propagation.")
            self.__backprop(target, data)
            """

        print(f"Layer weights are :\n{self.layer_weights}")
        print(f"Bias weights are :\n{self.bias_weights}")

        return
