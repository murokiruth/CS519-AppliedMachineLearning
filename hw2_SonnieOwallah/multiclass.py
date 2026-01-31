import numpy as np
from sgd import SGD

class MulticlassSGD:
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta   ##learning rate (how fast the model learns)
        self.n_iter = n_iter  ##number of iterations (epochs) for train
        self.random_state = random_state  ##random number generation  to ensure reproducibility when initializing weights.

    def fit(self, X, y):
        self.classes_ = np.unique(y)  #getting unique class labels
        self.classifiers = []  #list to store binary classifiers

        for cls in self.classes_:
            #creating a binary label vector for the current class
            y_binary = np.where(y == cls, 1, -1)
            
            #training a binary SGD classifier
            classifier = SGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            classifier.fit(X, y_binary)
            
            #storing the trained classifier
            self.classifiers.append(classifier)
        return self

    def predict(self, X):
        #getting decision scores for each class
        scores = np.array([classifier.net_input(X) for classifier in self.classifiers])
        
        #predicting the class with the highest score
        return self.classes_[np.argmax(scores, axis=0)]