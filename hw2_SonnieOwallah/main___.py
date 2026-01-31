import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from adaline import Adaline
from sgd import SGD
from multiclass import MulticlassSGD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx'
##https://archive.ics.uci.edu/dataset/247/istanbul+stock+exchange##

#loading Iris dataset
def load_iris_dataset():
    df1 = pd.read_csv('iris.data', header=None)   
    print(df1.head())
    positive_class = 'Iris-setosa'    ##defining the positive class
    y = df1.iloc[:, 4].values
    y = np.where(y == positive_class, 1, -1)   ##positive class: 1, Others(negative classes): -1
    X = df1.iloc[:, [0,2]].values   ##selecting two features for visualization, sepal length and petal length
    return X, y


#loading Istanbul Stock Exchange dataset
def load_istanbul_dataset():
    df2 = pd.read_excel(url, engine="openpyxl", header=1) ##skipping the first row (header)
    print("Dataset Information:")
    print(df2.head())
    print(f"Number of Rows: {df2.shape[0]}")
    print(f"Number of Features: {df2.shape[1] - 1}")  # exclude the target column
    #print(f"Classes: {np.unique(df2.iloc[:, -1])}")  # unique values in the target column
    df2 = df2.drop(columns=['date'])  # Drop the 'date' column
    X = df2.iloc[:, :-1].values   #features (all columns except the last)
    y = df2.iloc[:, -1].values    #labels (last column)
    y = pd.to_numeric(y, errors='coerce')  #convert to numeric values, set invalid values to NaN
    df2.dropna(inplace=True)  # Drop rows with NaN values
    y = np.where(y > 0, 1, -1) 
    return X, y

def main(classifier_name, dataset_name):
    if dataset_name == 'iris':
        X, y = load_iris_dataset()
    elif dataset_name == 'istanbul':
        X, y = load_istanbul_dataset()
    else:
        raise ValueError("Dataset name invalid. Use 'iris' or 'istanbul'.")


    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if classifier_name == 'multiclass_sgd':
        # training the multiclass classifier
        classifier = MulticlassSGD(eta=0.01, n_iter=10, random_state=1)
        
        # measuring training time
        start_time = time.time()
        classifier.fit(X_scaled, y)
        end_time = time.time()
        print(f"Multiclass SGD Training Time: {end_time - start_time:.4f} seconds")

        # making predictions on the training data
        y_pred = classifier.predict(X_scaled)

        # calculating final accuracy
        f_accuracy = accuracy_score(y, y_pred) * 100
        print(f"Multiclass SGD Final Accuracy: {f_accuracy:.2f}%")

        # plotting accuracy for each epoch (not applicable for multiclass)
        print("Accuracy plots are not available for multiclass classifiers.")
    else:
        raise ValueError("Classifier name invalid. Use 'multiclasssgd'.")
        

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ##testing different learning rates
    learning_rates = [0.001, 0.01, 0.1]    
    for eta in learning_rates:
        if classifier_name == 'perceptron':
            classifier = Perceptron(eta=0.1, n_iter=10)
        elif classifier_name == 'adaline':
            classifier = Adaline(eta=0.01, n_iter=10)
        elif classifier_name == 'sgd':
            classifier = SGD(eta=0.01, n_iter=10, random_state=1)
        else:
            raise ValueError("Classifier name invalid. Use 'perceptron', 'adaline' or 'sgd'.")

        # training classifier
        classifier.fit(X_scaled, y)
        
        #measuring training time
        start_time = time.time()
        end_time = time.time()
        print(f"{classifier_name.capitalize()} Learning Rate Training Time (eta={eta}): {end_time - start_time:.4f} seconds")

        # plotting training for different learning rates
    if classifier_name == 'perceptron':
        plt.plot(range(1, len(classifier.errors_) +1), classifier.errors_, marker='o', label=f'eta={eta}')
        plt.xlabel("Epochs")
        plt.ylabel("Number of Updates")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates {dataset_name.capitalize()} Dataset.")
        plt.show()
    elif classifier_name == 'adaline':
        plt.plot(range(1, len(classifier.losses_) + 1), classifier.losses_, marker='o', label=f'eta={eta}')
        plt.xlabel("Epochs")
        plt.ylabel("Cost (MSE)")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates {dataset_name.capitalize()} Dataset.")
        plt.show()
    elif classifier_name == 'sgd':
        plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o', label=f'eta={eta}')
        plt.xlabel("Epochs")
        plt.ylabel("Cost (MSE)")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates {dataset_name.capitalize()} Dataset.")
        plt.show()
    else:
        raise ValueError("Classifier name invalid. Use 'perceptron', 'adaline' or 'sgd'.")   
    
#--------------------------------------------------------------------------------

    ##testing different numbers of iterations
    n_iters = [10, 50, 100]    
    for n_iter in n_iters:
        if classifier_name == 'perceptron':
            classifier = Perceptron(eta=0.01, n_iter=n_iter)
        elif classifier_name == 'adaline':
            classifier = Adaline(eta=0.01, n_iter=n_iter)
        elif classifier_name == 'sgd':
            classifier = SGD(eta=0.01, n_iter=n_iter, random_state=1)
        else:
            raise ValueError("Classifier name invalid. Use 'perceptron', 'adaline' or 'sgd'.")    

    # training classifier
    classifier.fit(X_scaled, y)

    # measuring training time
    start_time = time.time()
    end_time = time.time()
    print(f"{classifier_name.capitalize()} Iterations Training Time (n_iter={n_iter}): {end_time - start_time:.4f} seconds")

    # plotting training for different iterations
    if classifier_name == 'perceptron':
        plt.plot(range(1, len(classifier.errors_) +1), classifier.errors_, marker='o', label=f'n_iter={n_iter}')
        plt.xlabel("Epochs")
        plt.ylabel("Number of Updates")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates {dataset_name.capitalize()} Dataset.")
        plt.show()
    elif classifier_name == 'adaline':
        plt.plot(range(1, len(classifier.losses_) + 1), classifier.losses_, marker='o', label=f'n_iter={n_iter}')
        plt.xlabel("Epochs")
        plt.ylabel("Cost (MSE)")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates {dataset_name.capitalize()} Dataset.")
        plt.show()
    elif classifier_name == 'sgd':
        plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o', label=f'n_iter={n_iter}')
        plt.xlabel("Epochs")
        plt.ylabel("Cost (MSE)")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates {dataset_name.capitalize()} Dataset.")
        plt.show()
    else:
        raise ValueError("Classifier name invalid. Use 'perceptron', 'adaline' or 'sgd'.")    

#--------------------------------------------------------------------------------
    # training classifier with default parameters
    if classifier_name == 'perceptron':
        classifier = Perceptron(eta=0.1, n_iter=10)
    elif classifier_name == 'adaline':
        classifier = Adaline(eta=0.01, n_iter=10)
    elif classifier_name == 'sgd':
        classifier = SGD(eta=0.01, n_iter=10, random_state=1)
    else:
        raise ValueError("Classifier name invalid. Use 'perceptron', 'adaline' or 'sgd'.")

    # training classifier
    classifier.fit(X_scaled, y)


    # making predictions on the training data
    y_pred = classifier.predict(X_scaled)

    #measuring training time
    start_time = time.time()
    end_time = time.time()
    print(f"{classifier_name.capitalize()} Training Time (default): {end_time - start_time:.4f} seconds")

    # calculating final accuracy
    f_accuracy = accuracy_score(y, y_pred) * 100
    print(f"{classifier_name.capitalize()} Final Accuracy: {f_accuracy:.2f}%")

    # plotting accuracy for each epoch
    plt.plot(range(1, len(classifier.accuracy_) + 1), classifier.accuracy_, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{classifier_name.capitalize()} Accuracy on {dataset_name.capitalize()} Dataset")
    plt.show()

    # plotting
    if classifier_name == 'perceptron':
        plt.plot(range(1, len(classifier.errors_) +1), classifier.errors_, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Number of Updates")
        plt.title(f"{classifier_name.capitalize()} Training on {dataset_name.capitalize()} Dataset.")
        plt.show()
    elif classifier_name == 'adaline':
        plt.plot(range(1, len(classifier.losses_) + 1), classifier.losses_, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Cost (MSE)")
        plt.title(f"{classifier_name.capitalize()} Training on {dataset_name.capitalize()} Dataset.")
        plt.legend()
        plt.show()
    elif classifier_name == 'sgd':
        plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Cost (MSE)")
        plt.title(f"{classifier_name.capitalize()} Training on {dataset_name.capitalize()} Dataset.")
        plt.legend()
        plt.show()
    else:
        raise ValueError("Classifier name invalid. Use 'perceptron', 'adaline' or 'sgd'.")    

        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <classifier_name> <dataset_name>")
        print("Example: python main.py perceptron iris")
        sys.exit(1)

    classifier_name = sys.argv[1]
    dataset_name = sys.argv[2]
    main(classifier_name, dataset_name)        




