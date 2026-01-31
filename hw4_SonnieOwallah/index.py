import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_moons

# datasets function
def load_dataset(name):
    if name == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        # displaying the shape of data and labels
        print(f"Iris Data shape: {X.shape}")
        print(f"Iris Labels shape: {y.shape}")
        # displaying the first few labels
        print(f"Iris First 10 labels: {y[:10]}")
        

    elif name == 'mnist':
        mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype(float)
        y = mnist.target.astype(int)
        # displaying the shape of data and labels
        print(f"MNIST Data shape: {X.shape}")
        print(f"MNIST Labels shape: {y.shape}")
        # displaying the first few labels
        print(f"MNIST First 10 labels: {y[:10]}")
        
        # creating the MNIST subset using train-test-split method with the stratify option (stratify=yes)
        X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.97, random_state=1, stratify=y)
        print(f"MNIST Data shape after subset: {X_subset.shape}")
        print(f"MNIST Labels shape after subset: {y_subset.shape}")
        print(f"MNIST First 10 labels after subset: {y_subset[:10]}")
        X = X_subset
        y = y_subset    
        
    else:
        raise ValueError("Dataset name invalid. Use 'iris' or 'mnist'.")
        
    return X, y

def reduction_analysis(dataset_name):
    X, y = load_dataset(dataset_name)
    # splitting data into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # standardize  the features
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # PCA
    start_pca = time.time()
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    end_pca = time.time()
    pca_time = end_pca - start_pca

    # printing explained variance ratio
    print(f"{dataset_name} Explained variance ratio:", pca.explained_variance_ratio_)


    # LDA
    start_lda = time.time()
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)
    end_lda = time.time()
    lda_time = end_lda - start_lda

    # Kernel PCA
    start_kcpa = time.time()
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    end_kcpa = time.time()
    kcpa_time = end_kcpa - start_kcpa

    # printing Execution Times
    print(f"{dataset_name} PCA Execution  Time: {pca_time}" )
    print(f"{dataset_name} LDA Execution  Time: {lda_time}")
    print(f"{dataset_name} KPCA Execution  Time: {kcpa_time}")

    # DecisionTreeClassifier
    criteria = ['gini', 'entropy']
    max_depths = [2, 4, 6]

    for criterion in criteria:
        for max_depth in max_depths: 
            tree_model = DecisionTreeClassifier (criterion=criterion, max_depth=max_depth, random_state=1)

        # fitting and predicting pca, lda and kpca
        start_dt = time.time()
        tree_model.fit(X_train_pca, y_train)
        y_pred_pca = tree_model.predict(X_test_pca)
        pca_accuracy = accuracy_score(y_pred_pca, y_test)

        tree_model.fit(X_train_lda, y_train)
        y_pred_lda = tree_model.predict(X_test_lda)
        lda_accuracy = accuracy_score(y_pred_lda, y_test)

        tree_model.fit(X_train_kpca, y_train)
        y_pred_kpca = tree_model.predict(X_test_kpca)
        kpca_accuracy = accuracy_score(y_pred_kpca, y_test)

        end_dt = time.time()
        dt_time = end_dt - start_dt
        print(f"{dataset_name} DT (criterion={criterion}, max_depth={max_depth}) Execution Time: {dt_time:.4f}")



        # calculating accuracy score
        print(f"{dataset_name} DT + PCA accuracy: {pca_accuracy:.4f}")
        print(f"{dataset_name} DT + LDA accuracy: {lda_accuracy:.4f}")
        print(f"{dataset_name} DT + KPCA accuracy: {kpca_accuracy:.4f}")

        # classification reports
        print(f"{dataset_name} Classification Report for PCA: {classification_report(y_test, y_pred_pca, zero_division=0)}")
        print(f"{dataset_name} Classification Report for LDA: {classification_report(y_test, y_pred_lda, zero_division=0)}")
        print(f"{dataset_name} Classification Report for Kernel PCA: {classification_report(y_test, y_pred_kpca, zero_division=0)}")


    #### Plotting 
        # PCA - Explained variance ratio Vs Principal components
    plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, align='center', label='Individual Explained Variance') 
    plt.step(range(1,len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative Explained Variance') 
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components') 
    plt.title(f"{dataset_name} PCA - Explained variance ratio Vs Principal components")
    plt.legend(loc='best') 
    plt.tight_layout()
    plt.show()

        # function for plotting the data after dimensionality reduction
    def plot_transformed_data(X_transformed, y, title):
        for label in np.unique(y):
            plt.scatter(X_transformed[y == label, 0], X_transformed[y == label, 1], label=f"Class {label}", alpha=0.7)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{dataset_name} {title}")
        plt.legend()
        plt.grid()
        plt.show()

        # plotting PCA, LDA, and KPCA results
    plot_transformed_data(X_train_pca, y_train, "PCA Projection")
    plot_transformed_data(X_train_lda, y_train, "LDA Projection")
    plot_transformed_data(X_train_kpca, y_train, "Kernel PCA Projection")

        # plotting KPCA using make_moons dataset
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=1)
    X_moons_std = StandardScaler().fit_transform(X_moons)

    kpca_moons = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
    X_moons_kpca = kpca_moons.fit_transform(X_moons_std)

    plot_transformed_data(X_moons_kpca, y_moons, "Kernel PCA Projection (make_moons dataset)")  

reduction_analysis('iris')
reduction_analysis('mnist')    