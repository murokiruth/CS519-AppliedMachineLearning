Project Title: HW3: Compare classifiers in scikit-learn library.
Author: Sonnie Owallah
Date: 02/14/2025

Description: 
This Python program implements and evaluates the performance of six classifiers—Perceptron, Logistic Regression, Linear SVM, Non-Linear SVM (RBF), Decision Tree, and KNN—on two datasets; Digits Dataset: A built-in dataset from the scikit-learn library. Heart Disease Dataset: A dataset downloaded from Kaggle (https://www.kaggle.com/datasets/abdmental01/heart-disease-dataset?resource=download) . The program evaluates how tuning hyperparameters affects model accuracy and training/testing times. It also creates plots to show how hyperparameters influence performance.
System and Software Requirements:
1. Python version 3.10 or higher.
2. Pandas library
3. Matplotlib library
4. Scikit-learn library

Files Included:
1. A zip file named hw3_sonnieowallah which has the following files.
2. readme.txt - a file with detailed instructions on how to run the Python program.
3. main.py: The main Python script to execute the program.
4. heart_disease_cleaned.csv: The Heart Disease dataset (downloaded from Kaggle).
5. report.pdf

Steps to Run the Program:
1. Ensure you have Python 3.10 or higher. You can do this by opening your command prompt/terminal and running  python --version. If you do not have Python or need to update the version, visit the official Python website at https://www.python.org/downloads/.
2. Install the Pandas library from the command prompt/terminal using  pip install pandas .
3. Install the Matplotlib library from the command prompt/terminal using  pip install matplotlib .
4. Install the Scikit-learn library from the command prompt/terminal using  pip install scikit-learn.
5. Extract the files in the zip file and save them in an unzipped directory/folder. 
6. Ensure that all the files are in the same directory.
7. To run the program from the command prompt/ terminal, open your command prompt/ terminal and navigate to the directory containing the files. 
8. Execute the following commands:    python main.py 

Expected Output:
1) Text output
* Dataset preview (first few rows).
* Number of rows and features in each dataset.
* Training and testing accuracy for each classifier.
* Training and testing times for each classifier.
* Best hyperparameters for each classifier.

2) Generated plots
* Perceptron: Training Accuracy vs. Learning Rate (eta0).
* Logistic Regression: Training and Testing Accuracy vs. Regularization Strength (C).
* Linear SVM: Training Accuracy vs. Regularization Strength (C).
* Non-Linear SVM (RBF): Training Accuracy vs. Gamma.
* Decision Tree: Training Accuracy vs. Maximum Depth (max_depth).
* KNN: Training and Testing Accuracy vs. Number of Neighbors (n_neighbors).

Troubleshooting: 
- If the script fails to execute, ensure the Python version is 3.10 or higher, the required packages are installed, and verify the path for the dataset is correct
