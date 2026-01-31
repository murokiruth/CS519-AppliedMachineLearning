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

# plotting
def plot_training(classifier_name, costs, label, plot_type):
    plt.figure()
    plt.plot(range(1, len(costs) + 1), costs, marker='o', label=label)
    plt.xlabel("Epochs")
      

    if plot_type == "accuracy":
        plt.ylabel("Accuracy (%)") 
        plt.title(f"{classifier_name.capitalize()} Accuracy on {dataset_name.capitalize()} Dataset")
    elif plot_type == "iterations":
        plt.ylabel("Cost/Errors")
        plt.title(f"{classifier_name.capitalize()} Training for Different Iterations ({dataset_name.capitalize()} Dataset)") # More specific title
    elif plot_type == "learning_rate":
        plt.ylabel("Cost/Errors")
        plt.title(f"{classifier_name.capitalize()} Training for Different Learning Rates ({dataset_name.capitalize()} Dataset)") # More specific title
    else:
        plt.ylabel("Cost/Errors")
        plt.title(f"{classifier_name.capitalize()} Training ({dataset_name.capitalize()} Dataset)")

    plt.legend()
    plt.show()

# defining classifiers

classifiers = {
        'perceptron': Perceptron,
        'adaline': Adaline,
        'sgd': SGD,
        'multiclass_sgd': MulticlassSGD
    }

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

    
    classifier_name_lower = classifier_name.lower()

    if classifier_name_lower not in classifiers:
        raise ValueError("Invalid classifier name.")
    
    classifier_class = classifiers[classifier_name_lower]

    if classifier_name_lower == 'multiclass_sgd':
        classifier = classifier_class(eta=0.01, n_iter=10, random_state=1)
        start_time = time.time()
        classifier.fit(X_scaled, y)  # Train on the entire dataset
        end_time = time.time()
        print(f"Multiclass SGD Training Time: {end_time - start_time:.4f} seconds")

        y_pred = classifier.predict(X_scaled)  # Predict on the entire dataset
        f_accuracy = accuracy_score(y, y_pred) * 100  # Evaluate on the entire dataset
        print(f"Multiclass SGD Final Accuracy: {f_accuracy:.2f}%")
    else:
        learning_rates = [0.001, 0.01, 0.1]
        n_iters = [10, 50, 100]

        for eta in learning_rates:
            for n_iter in n_iters:
                classifier = classifier_class(eta=eta, n_iter=n_iter, random_state=1 if classifier_name_lower == 'sgd' else None)

                start_time = time.time()
                classifier.fit(X_scaled, y)  # Train on the entire dataset
                end_time = time.time()
                print(f"{classifier_name.capitalize()} Training Time (eta={eta}, n_iter={n_iter}): {end_time - start_time:.4f} seconds")

                y_pred = classifier.predict(X_scaled)  # Predict on the entire dataset
                f_accuracy = accuracy_score(y, y_pred) * 100  # Evaluate on the entire dataset
                print(f"{classifier_name.capitalize()} Final Accuracy (eta={eta}, n_iter={n_iter}): {f_accuracy:.2f}%")

                if classifier_name in ('perceptron', 'adaline', 'sgd'):
                    costs = classifier.errors_ if classifier_name == 'perceptron' else classifier.losses_ if classifier_name == 'adaline' else classifier.cost_
                    plot_training(classifier_name, costs, f'eta={eta}, n_iter={n_iter}', plot_type="learning_rate")


        # plotting for different iterations (after the loop)
        for n_iter in n_iters:
            classifier = classifier_class(eta=0.01, n_iter=n_iter, random_state=1 if classifier_name_lower == 'sgd' else None)
            classifier.fit(X_scaled, y)
            costs = classifier.errors_ if classifier_name == 'perceptron' else classifier.losses_ if classifier_name == 'adaline' else classifier.cost_
            plot_training(classifier_name, costs, f'n_iter={n_iter}', plot_type="iterations") # Correct plot type

        # plotting with default parameters (after the loop)
        classifier = classifier_class(eta=0.01, n_iter=10, random_state=1 if classifier_name_lower == 'sgd' else None)
        classifier.fit(X_scaled, y)
        costs = classifier.errors_ if classifier_name == 'perceptron' else classifier.losses_ if classifier_name == 'adaline' else classifier.cost_
        plot_training(classifier_name, costs, "Default Parameters", plot_type=None)  # Default plot type

        # plotting Accuracy
        plot_training(classifier_name, classifier.accuracy_, "Accuracy", plot_type="accuracy") # Accuracy plot

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <classifier_name> <dataset_name>")
        print("Example: python main.py perceptron iris")
        sys.exit(1)

    classifier_name = sys.argv[1]
    dataset_name = sys.argv[2]
    main(classifier_name, dataset_name)        



