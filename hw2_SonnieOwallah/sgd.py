import numpy as np
from sklearn.metrics import accuracy_score


class SGD:
    ### defining the constructor to initialize SGD
    def __init__(self, eta = 0.01, n_iter = 100, shuffle = True, random_state = 1):
        self.eta = eta   ##learning rate (how fast the model learns)
        self.n_iter = n_iter   ##number of iterations (epochs) for training
        self.random_state = random_state   ##random number generation
        self.shuffle = shuffle   ##checking whether to shuffle data before each epoch
        self.w_initialized = False   ##checking if weights are initialized

    ### training the classifier using Stochastic Gradient Descent
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])   ##initializing weights
        self.cost_ = [] ##list for storing the cost (sum of squared errors) for each epoch
        self.accuracy_ = []  #list for storing accuracy for each epoch

        for i in range(self.n_iter):
            if self.shuffle:  
                X, y = self._shuffle(X, y)   ##shuffling data
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))  ##updating weights for each sample
            avg_cost = np.mean(cost)   ##calculating average cost for the epoch
            self.cost_.append(avg_cost)   ##storing the cost

            ##calculating accuracy
            y_pred = self.predict(X)
            acc = accuracy_score(y, y_pred) *100
            self.accuracy_.append(acc)

            ##checking convergence
            if i > 0 and abs(self.cost_[i] - self.cost_[i-1]) < 1e-5:
                print(f"SGD converged at epoch {i+1}")
                break

        return self  


    ###fit training without reinitializing weights
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])    ##initializing weights if not already done
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,y)  ##updating weights for a single sample

        return self
    
    ###initializing weights    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)  ##initializing a random number generator
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)  ##initializing weights 
        self.b_ = np.float64(0.0)  ##initializing bias
        self.w_initialized = True
             

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))    ##generating shuffled indices
        return X[r], y[r]     ##returning shuffled data
    
    def _update_weights(self, xi, target):
        output = self.net_input(xi)   ##calculating net input
        error = (target - output)   ##calculating error
        self.w_ += self.eta * xi.dot(error)  ##updating weights
        self.b_ += self.eta * error   ##updating bias
        cost = 0.5 * error**2  ##calculating cost for this sample
        return cost

    ##calculating netinput, the weighted sum of inputs (dot product of weights and inputs) + bias
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    ##linear activation function
    def activation(self, X):
        return X

    ##making predictions using the trained
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)