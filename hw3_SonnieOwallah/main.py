import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
def load_dataset(name):
    if name == 'digits':
        data1 = datasets.load_digits()
        X = data1.data
        y = data1.target
        print("Digits Dataset Information:")
        print(f"Number of Rows: {X.shape[0]}")
        print(f"Number of Features: {X.shape[1]}")
    elif name == 'heart disease':
        data2 = pd.read_csv("heart_disease_cleaned.csv")
        categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope','thal'] 
        data2 = pd.get_dummies(data2, columns=categorical_cols)
        X = data2.iloc[:, :-1].values  # Features after converting all columns with categorical features into numerical representations
        y = data2.iloc[:, -1].values  # Labels (the last column, which is the target)
        print("Heart Disease Information:")
        print(data2.head())
        print(f"Number of Rows: {X.shape[0]}")
        print(f"Number of Features: {X.shape[1]}")
    else:
        raise ValueError("Dataset name invalid. Use 'digits' or 'heart disease'.")
    return X, y

# plotting function
def plot_training(classifier_name, dataset_name, x_values, y_values, x_label, y_label, title, plot_type):    
    plt.figure()
    if plot_type == 'training':
        plt.plot(x_values, y_values, marker='D', label=classifier_name)
    elif plot_type == 'testing':
        plt.bar(x_values, y_values, label=classifier_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title} ({dataset_name} Dataset)")
    plt.legend()
    plt.show()
    

# defining classifiers
classifiers = {
        "Perceptron": Perceptron(eta0=0.1,max_iter=100, random_state=1),
        "Logistic Regression": OneVsRestClassifier(LogisticRegression(C=100.0, solver='liblinear', max_iter=100)),
        "Linear SVM": SVC(kernel='linear', C=1.0, random_state=1),
        "Non-Linear SVM (RBF)": SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0),
        "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    }


# running classifier and report performance
def main(classifier_name, dataset_name):
    X, y = load_dataset(dataset_name)

    ##splitting training and test datasets (80:20) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    print('Labels count in y:', np.bincount(y))
    print('Labels count in y_train:', np.bincount(y_train))
    print('Labels count in y_test:', np.bincount(y_test))


# Feature scaling
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    classifier = classifiers[classifier_name]


# tuning one hyperparameter per classifier
    param_grid = {}
    if classifier_name == 'Perceptron':
        param_grid = {'eta0': [0.01, 0.1, 1]}   #tuning learning rate for Perceptron
    elif classifier_name == 'Logistic Regression':
        param_grid = {'estimator__C': [0.1, 1, 10]}      #tuning regularization strength for Logistic Regression
    elif classifier_name == 'Linear SVM':
        param_grid = {'C': [1, 10, 100]}       #tuning regularization strength for Linear SVM
    elif classifier_name == 'Non-Linear SVM (RBF)':
        param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}   #tuning parameters for Non-Linear SVM
    elif classifier_name == 'Decision Tree':
        param_grid = {'max_depth': [3, 5, 10]}    #tuning maximum depth of the tree
    elif classifier_name == 'KNN':
        param_grid = {'n_neighbors': [5, 7, 9]}      #tuning number of neighbors to consider
    
    best_params = classifier.get_params()
    if param_grid:
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')    #searching for the best parameters
        grid_search.fit(X_train, y_train)       #training the model with different parameters
        classifier = grid_search.best_estimator_    #using the best model found
        best_params = grid_search.best_params_      #saving the best parameters
    else:
        classifier.fit(X_train_std, y_train)        #if no tuning, just training the model

    # measuring performance
        #training the model 
    train_start = time.time()
    classifier.fit(X_train_std, y_train)        #training the model
    train_time = time.time() - train_start  #calculating training time
    
        #testing the model 
    test_start = time.time()
    y_pred = classifier.predict(X_test_std)     #making predictions on the test set
    test_time = time.time() - test_start    #calculating testing time
    
    # calculating accuracy on the training and testing sets
    y_train_pred = classifier.predict(X_train_std)
    train_accuracy = accuracy_score(y_train, y_train_pred)      #accuracy on training data
    test_accuracy = accuracy_score(y_test, y_pred)      #accuracy on testing data
    
    # printing results
    print(f"{classifier_name.capitalize()}: Training Accuracy={train_accuracy:.4f}, Testing Accuracy={test_accuracy:.4f}, Training Time={train_time:.4f}s, Testing Time={test_time:.4f}s")
    print(f"Best Hyperparameter: {best_params}\n")


    # a plot for each classifier
    ## Perceptron plot: Training Accuracy vs. eta0
    if classifier_name == 'Perceptron':
        eta0_values = [0.01, 0.1, 1]
        accuracy_list = []
        for eta0 in eta0_values:
            clf = Perceptron(eta0=eta0, random_state=1, max_iter=1000)
            clf.fit(X_train_std, y_train)
            y_train_pred = clf.predict(X_train_std)
            accuracy_list.append(accuracy_score(y_train, y_train_pred))
        plot_training(classifier_name, dataset_name, eta0_values, accuracy_list, "Learning Rate (eta0)", "Accuracy (%)", f"{classifier_name.capitalize()} Training Accuracy vs Learning Rate", 'training')
    
    ## Logistic Regression: Training & Testing Accuracy vs. Regularization Strength (C)
    elif classifier_name == 'Logistic Regression':
        C_values = [0.1, 1, 10]
        train_accuracy_list = []
        test_accuracy_list = []
        for C in C_values:
            clf = OneVsRestClassifier(LogisticRegression(C=C, solver='liblinear', max_iter=1000))
            clf.fit(X_train_std, y_train)
            y_train_pred = clf.predict(X_train_std)
            y_test_pred = clf.predict(X_test_std)
            train_accuracy_list.append(accuracy_score(y_train, y_train_pred))
            test_accuracy_list.append(accuracy_score(y_test, y_test_pred))
        plot_training(classifier_name, dataset_name, C_values, train_accuracy_list, "Regulation Strength (C)", "Accuracy (%)", f"{classifier_name.capitalize()} Training Accuracy vs C ", 'training')
        plot_training(classifier_name, dataset_name, C_values, test_accuracy_list, "Regulation Strength (C)", "Accuracy (%)", f"{classifier_name.capitalize()} Testing Accuracy vs C ", 'testing')


    ## Linear SVM: Training Accuracy vs. Regularization Strength (C)
    elif classifier_name == 'Linear SVM':
        C_values = [1, 10, 100]
        accuracy_list = []
        for C in C_values:
            clf = SVC(kernel='linear', C=C, random_state=1)
            clf.fit(X_train_std, y_train)
            y_train_pred = clf.predict(X_train_std)
            accuracy_list.append(accuracy_score(y_train, y_train_pred))
        plot_training(classifier_name, dataset_name, C_values, accuracy_list, "Regulation Strength (C)", "Accuracy (%)", f"{classifier_name.capitalize()} Training Accuracy vs C ", 'training')


    ## Non-Linear SVM (RBF): Training Accuracy vs. Gamma (gamma)
    elif classifier_name == 'Non-Linear SVM (RBF)':
        gamma_values = [0.01, 0.1, 1]
        accuracy_list = []
        for gamma in gamma_values:
            clf = SVC(kernel='rbf', random_state=1, gamma=gamma)
            clf.fit(X_train_std, y_train)
            y_train_pred = clf.predict(X_train_std)
            accuracy_list.append(accuracy_score(y_train, y_train_pred))
        plot_training(classifier_name, dataset_name, gamma_values, accuracy_list, "Gamma", "Accuracy (%)", f"{classifier_name.capitalize()} Training Accuracy vs Gamma", 'training')


    ## Decision Tree: Training Accuracy vs. Maximum Depth (max_depth)
    elif classifier_name == 'Decision Tree':
        max_depth_values = [3, 5, 10]
        accuracy_list = []
        for depth in max_depth_values:
            clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=1)
            clf.fit(X_train_std, y_train)
            y_train_pred = clf.predict(X_train_std)
            accuracy_list.append(accuracy_score(y_train, y_train_pred))
        plot_training(classifier_name, dataset_name, max_depth_values, accuracy_list, "Max Depth", "Accuracy (%)", f"{classifier_name.capitalize()} Training Accuracy vs Max Depth", 'training')



    
     ## KNN: Training & Testing Accuracy vs. Number of Neighbors (n_neighbors)
    elif classifier_name == 'KNN':
        n_neighbors_values = [5, 7, 9]
        train_accuracy_list = []
        test_accuracy_list = []
        for n in n_neighbors_values:
            clf = KNeighborsClassifier(n_neighbors=n, p=2, metric='minkowski')
            clf.fit(X_train_std, y_train)
            y_train_pred = clf.predict(X_train_std)
            y_test_pred = clf.predict(X_test_std)
            train_accuracy_list.append(accuracy_score(y_train, y_train_pred))
            test_accuracy_list.append(accuracy_score(y_test, y_test_pred))
        plot_training(classifier_name, dataset_name, n_neighbors_values, train_accuracy_list, "Number of Neighbors(n_neighbors)", "Accuracy (%)",  f"{classifier_name.capitalize()} Training Accuracy vs Number of Neighbors ", 'training')
        plot_training(classifier_name, dataset_name, n_neighbors_values, test_accuracy_list, "Number of Neighbors(n_neighbors)", "Accuracy (%)", f"{classifier_name.capitalize()} Testing Accuracy vs Number of Neighbors ", 'testing')
        

     




# running classifiers on digits dataset
dataset_name = 'digits'
for classifier in classifiers:
    main(classifier, dataset_name)

# running Logistic Regression, Non-Linear SVM, and Decision Tree on Heart Disease dataset
dataset_name = 'heart disease'
for classifier in ['Logistic Regression', 'Non-Linear SVM (RBF)', 'Decision Tree']:
    main(classifier, dataset_name)



    
    
    