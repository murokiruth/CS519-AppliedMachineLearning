from sklearn import datasets
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

##loading digits datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target 

##splitting training and test datasets (80:20) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
print('Labels count in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))

##standard scaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

##training perceptron model
ppn = Perceptron(eta0=0.1,max_iter=40, random_state=1)
ppn.fit(X_train_std, y_train)

##making predictions
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d'%(y_test!=y_pred).sum())

##classification accuracy
print('Accuracy: %.3f' % accuracy_score(y_test,y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

##training logistic regression model
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

##logistic regression prediction
lr.predict_proba(X_test_std[:3, :])

##training support vector machine (SVM) model
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

##non-linear SVM using Radial Basis Func9on (RBF) kernel
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X,y)


####Building a decision tree
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)
tree.plot_tree(tree_model)
plt.show()

####RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)


####KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)


