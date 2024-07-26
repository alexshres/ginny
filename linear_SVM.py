import numpy as np
from scipy.optimize import minimize

class LinearSVM:
    """
    Linear SVM Model for now,
    need to add code to allow for kernel functions
    """

    def __init__(self, training_data, training_labels, 
                 epochs=10, regularization=1.0):

        self.X = training_data
        self.y = training_labels
        self.epochs = epochs
        self.C = regularization         # regularization parameters


        # Computing gram matrix
        self._gram = np.dot(self.X, self.X.T)

    def __objective(self, alpha, Q):
        return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)

    def __quadratic_solver(self):
        Q = np.outer(self.y, self.y) * K

        # Defining contraint for solver
        cons = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}

        pass


    

