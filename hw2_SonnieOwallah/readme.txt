Project Title: HW2: Single-layer Linear Neural Networks
Author: Sonnie Owallah
Date: 01/30/2025

Description: 
This Python program implements and evaluates Perceptron, Adaline, and SGD classification algorithms on the Iris or Istanbul Stock Exchange dataset. It explores how learning rates and training iterations affect performance, using StandardScaler for feature scaling. It generates plots for training progress (cost/errors), accuracy, and different settings. Users can specify the classifier and dataset via command-line arguments. Multiclass SGD is also included.
System and Software Requirements:
1. Python version 3.10 or higher.
2. Pandas library
3. Matplotlib library
4. Scikit-learn library

Files Included:
1. A zip file named hw2_sonnieowallah which has the following files.
2. readme.txt - a file with detailed instructions on how to run the Python program.
3. main.py - python script
4. adaline.py
5. sgd.py
6. multiclass.py
7. iris.data - Iris dataset
8. Instanbul stock exchange dataset url (dataset loaded from the web)

Steps to Run the Program:
Ensure you have Python 3.10 or higher. You can do this by opening your command prompt/terminal and running ‘py -3 –version’. If you do not have Python or need to update the version, visit the official Python website at https://www.python.org/downloads/.
Install the Pandas library from the command prompt/terminal using ‘pip install pandas’.
Install the Matplotlib library from the command prompt/terminal using ‘pip install matplotlib’.
Install the Scikit-learn library from the command prompt/terminal using ‘pip install scikit-learn.
Extract the files in the zip file and save them in an unzipped directory/folder. 
Ensure that all the files are in the same directory.
To run the program from the command prompt/ terminal, open your command prompt/ terminal and navigate to the directory containing the files. 
7) Execute the following commands:
* python main.py perceptron iris
* python main.py perceptron Istanbul
* python main.py adaline iris
* python main.py adaline Istanbul
* python main.py sgd iris
* python main.py sgd Istanbul


Expected Output:
1) Text output 
* Dataset preview (first few rows)
* Training time per classifier
* Accuracy and convergence details
* Number of rows and features

2) Generated plots
* Training progress (cost/errors)
* Accuracy vs. epochs
* Impact of different learning rates and iterations 


Troubleshooting: 
- If the script fails to execute, ensure the Python version is 3.10 or higher, the required packages are installed, and verify the path for the dataset is correct and internet connectivity (for the Istanbul dataset)
