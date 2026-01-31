import numpy as np
from sklearn.metrics import accuracy_score


class Perceptron:
    ### defining the constructor to initialize the perceptron
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta   ##learning rate (how fast the model learns)
        self.n_iter = n_iter  ##number of iterations (epochs) for train
        self.random_state = random_state  ##random number generation  to ensure reproducibility when initializing weights.

    ### training the perceptron on the input data (X) and labels (y)
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)  ##initializing a random number generator
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  ##initializing weights and bias randomly
        self.errors_ =[]   ##list for storing the number of misclassifications in each epoch/iteration
        self.accuracy_ = []  #list for storing accuracy for each epoch

        for i in range(self.n_iter):   ##looping for the specified number of epochs/iterations
            errors = 0   ##counter for number of misclassifications in the current epoch
            for xi, target in zip(X,y):    ##looping through each data point (xi) and its label (target)
                update = self.eta * (target - self.predict(xi))  ##calculating the update for weights and bias
                self.w_[1:] += update*xi   ##updating the weights 
                self.w_[0] += update   ##updating the bias 
                errors += int(update != 0.0)    ##incrementing error count if update is not zero (misclassification)
            self.errors_.append(errors)    ##storing the number of errors for the current epoch
  
            ##calculating accuracy
            y_pred = self.predict(X)
            acc = accuracy_score(y, y_pred) *100
            self.accuracy_.append(acc)

            ##checking convergence
            if i > 0 and abs(self.errors_[i] - self.errors_[i-1]) < 1e-5:
                print(f"Perceptron converged at epoch {i+1}")
                break
           
        return self    ##returning the trained model

    ### prediction
        ##computing the weighted sum of inputs (dot product of weights and inputs) + bias
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        ##making predictions using the trained Perceptron
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
