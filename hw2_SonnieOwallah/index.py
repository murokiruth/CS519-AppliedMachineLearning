import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from adaline import Adaline
from sgd import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx'
##https://archive.ics.uci.edu/dataset/247/istanbul+stock+exchange##

#loading Iris dataset
df1 = pd.read_csv('iris.data', header=None)   
#df2 = pd.read_excel(url, engine="openpyxl")
#print(df2.head())
positive_class = 'Iris-setosa'    ##defining the positive class
y = df1.iloc[:, 4].values
y = np.where(y == positive_class, 1, -1)   ##positive class: 1, Others(negative classes): -1
X = df1.iloc[:, [0,2]].values   ##selecting two features for visualization, sepal length and petal length

#loading Istanbul Stock Exchange dataset
df2 = pd.read_excel(url, engine="openpyxl")
print(df2.head())
X = df2.iloc[:, :-1].values
y = df2.iloc[:, -1].values
y = np.where(y > 0, 1, -1)   


##splitting training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

##check the mean values and the standard derivations before and after feature scaling
# X_train[:,0].mean()
# X_train[:,1].mean()
# X_train[:,0].std()
# X_train[:,1].std()

# X_train_std[:,0].mean()
# X_train_std[:,1].mean()
# X_train_std[:,0].std()
# X_train_std[:,1].std()
print("Before scaling - Mean:", X_train.mean(axis=0))  # Mean of each column
print("Before scaling - Std:", X_train.std(axis=0))    # Std of each column
print("After scaling - Mean:", X_train_std.mean(axis=0))  # Mean of each column
print("After scaling - Std:", X_train_std.std(axis=0))    # Std of each column


###testing the Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X_train_std, y_train)
    ##plotting the number of errors during training
plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Number of Updates")
plt.title("Perceptron Training Errors")
plt.show()


###testing the adaline model
ada = Adaline(eta=0.01, n_iter=10)
ada.fit(X_train_std, y_train)
plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o', label='Adaline')
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Adaline Training Costs")
plt.legend()
plt.show()

###testing the adaline model
sgd = SGD(eta=0.01, n_iter=10, random_state=1)
sgd.fit(X_train_std, y_train)
plt.plot(range(1, len(sgd.cost_) + 1), sgd.cost_, marker='o', label='SGD')
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("SGD Training Costs")
plt.legend()
plt.show()


