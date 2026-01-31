import numpy as np
from sklearn.metrics import accuracy_score


class Adaline:
    ### defining the constructor to initialize adaline
    def __init__(self, eta = 0.01, n_iter = 100, random_state = 1):
        self.eta = eta   ##learning rate (how fast the model learns)
        self.n_iter = n_iter   ##number of iterations (epochs) for training
        self.random_state = random_state   ##random number generation

    ### training the adaline on the input data (X) and labels (y)#
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)   ##initializing a random number generator
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])  ##initializing weights and bias randomly
        self.b_ = np.float64(0.0)  ##initializing bias
        self.losses_ = []  ##list for storing the cost (sum of squared errors) for each epoch
        self.accuracy_ = []  #list for storing accuracy for each epoch

        for i in range(self.n_iter):  ##looping for the specified number of epochs
            net_input = self.net_input(X)    ##computing net input
            output = self.activation(net_input)     ##computing linear activation (identity function for Adaline)
            errors = (y - output)   ##calculating the difference between actual and predicted value
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]   ##updating the weights using gradient descent
            self.b_ += self.eta * errors.mean()   ##updating the bias term
            loss = (errors ** 2).mean()   ##calculating the loss
            self.losses_.append(loss)   ##storing the loss for the current epoch

            ##calculating accuracy
            y_pred = self.predict(X)
            acc = accuracy_score(y, y_pred) *100
            self.accuracy_.append(acc)

            ##checking convergence
            if i > 0 and abs(self.losses_[i] - self.losses_[i-1]) < 1e-5:
                print(f"Adaline converged at epoch {i+1}")
                break

        return self ##returning the trained model
    

    ##calculating netinput, the weighted sum of inputs (dot product of weights and inputs) + bias
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    ##linear activation function
    def activation(self, X):
        return X

    ##making predictions using the trained
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)