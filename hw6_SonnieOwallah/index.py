import numpy as np
import pandas as pd
import time
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from matplotlib import cm
from collections import Counter

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
        # displaying the number of distinct labels and instances of each label
        print(f"Iris Number of distinct labels: {len(np.unique(y))}")
        label_counts = Counter(y)
        print(f"Iris Number of instances per label: {label_counts}")
        

    elif name == 'mnist':
        mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype(float)
        y = mnist.target.astype(int)
        # displaying the shape of data and labels
        print(f"MNIST Data shape: {X.shape}")
        print(f"MNIST Labels shape: {y.shape}")
        # displaying the first few labels
        print(f"MNIST First 10 labels: {y[:10]}")
        # displaying the number of distinct labels and instances of each label
        print(f"MNIST Number of distinct labels: {len(np.unique(y))}")
        label_counts = Counter(y)
        print(f"MNIST Number of instances per label: {label_counts}")
        
        # creating the MNIST subset using train-test-split method with the stratify option (stratify=yes)
        X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.97, random_state=1, stratify=y)
        print(f"MNIST Data shape after subset: {X_subset.shape}")
        print(f"MNIST Labels shape after subset: {y_subset.shape}")
        print(f"MNIST First 10 labels after subset: {y_subset[:10]}")
        # displaying the number of distinct labels and instances of each label in subset
        print(f"MNIST Subset Number of distinct labels: {len(np.unique(y_subset))}")
        subset_label_counts = Counter(y_subset)
        print(f"MNIST Subset Number of instances per label: {subset_label_counts}")
        X = X_subset
        y = y_subset    
        
    else:
        raise ValueError("Dataset name invalid. Use 'iris' or 'mnist'.")
        
    return X, y

def KMeans_clustering(dataset_name, X, y):
    start_KMeans = time.time()
    # initializing KMeans with true number of classes
    km = KMeans(n_clusters = len(np.unique(y)), 
                init='random',
                n_init=10, 
                max_iter=300,
                tol=1e-04, 
                random_state=0)
    y_km = km.fit_predict(X)  # performing clustering
    end_KMeans = time.time()
    KMeans_time = end_KMeans - start_KMeans

    # converting labels to integers for consistency
    y_km = np.array(y_km, dtype=int)
    
    # calculating metrics
    sse = km.inertia_
    silhouette_avg = silhouette_score(X, y_km)
    ari = adjusted_rand_score(y, y_km)
    
    # printing results
    print("\nK-means Results:")
    print(f"Cluster labels: {np.unique(y_km)}")
    print(f"SSE: {sse:.3f}")
    print(f"Silhouette Coefficient: {silhouette_avg:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Execution Time: {KMeans_time:.3f} seconds")

    # handling cluster centers based on dataset
    if dataset_name == 'iris':
        print(f"Cluster centers:\n{km.cluster_centers_}")
    else:
        np.save(f"{dataset_name}_cluster_centers.npy", km.cluster_centers_)
        print(f"Cluster centers saved to {dataset_name}_cluster_centers.npy (not printed).")


    return y_km
# ELBOW METHOD
def elbow_method(dataset_name, X):
    distortions = []    # store SSE values
    silhouette_scores = []  # store silhouette scores
    K_values = range(2, 11)  # start from 2 clusters for silhouette
    for i in K_values:  
        km = KMeans(n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, km.labels_))
    
    # creating dual plot:
    # plotting elbow method
    #plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_values, distortions, marker='o', color='blue')
    plt.xlabel('Number of Clusters(K)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title(f'Elbow Method for {dataset_name}')
    
    # plotting silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(K_values, silhouette_scores, marker='o', color='green')
    plt.xlabel('Number of clusters(K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Analysis for {dataset_name}')
    plt.tight_layout()
    plt.show()

    # finding optimal K
    optimal_k = K_values[np.argmin(np.diff(distortions))]
    print(f"The Optimal K selected from the Elbow Method for {dataset_name}: {optimal_k}")

def Scipy_hierarchical_clustering(dataset_name, X, y):
    start_Scipy = time.time()
    # creating linkage matrix using Ward's method
    row_clusters = linkage(X, method='ward', metric='euclidean')
    end_Scipy = time.time()
    Scipy_time = end_Scipy - start_Scipy
    
    # getting cluster labels
    k = len(np.unique(y))  # use true number of classes
    clusters = fcluster(row_clusters, k, criterion='maxclust') - 1  
    
    # calculating metrics
    silhouette_avg = silhouette_score(X, clusters)
    ari = adjusted_rand_score(y, clusters)
    
    print("\nSciPy Hierarchical Results:")
    print(f"Linkage matrix (first 10 rows):\n{row_clusters[:10]}")
    print(f"Cluster labels: {clusters}")
    print(f"Silhouette Coefficient: {silhouette_avg:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Execution Time: {Scipy_time:.3f} seconds")

    # plotting dendrogram
    #plt.figure(figsize=(10, 5))
    dendrogram(row_clusters, truncate_mode='level', p=5)
    plt.ylabel('Euclidean distance')
    plt.title(f'Dendrogram for {dataset_name} (Complete linkage)')
    plt.tight_layout()
    plt.show()

    return clusters

def Sklearn_hierarchical_clustering(dataset_name, X, y):
    start_AC = time.time()
    ac = AgglomerativeClustering(
        n_clusters=len(np.unique(y)),  # use true number of classes
        metric='euclidean', 
        linkage='ward')
    ac_labels = ac.fit_predict(X)
    end_AC = time.time()
    AC_time = end_AC - start_AC
    
    # calculating metrics
    silhouette_avg = silhouette_score(X, ac_labels)
    ari = adjusted_rand_score(y, ac_labels)
    
    print("\nScikit-learn Hierarchical Results:")
    print(f"Cluster labels: {ac_labels}")
    print(f"Silhouette Coefficient: {silhouette_avg:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Execution Time: {AC_time:.3f} seconds")

    return ac_labels

def plot_silhouette(X, cluster_labels, dataset_name, algorithm_name):
    cluster_labels = np.array(cluster_labels, dtype=int)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        print(f"Cannot plot silhouette: only 1 cluster found in {algorithm_name} for {dataset_name}")
        return
    
    # calculating silhouette values
    silhouette_vals = silhouette_samples(X, cluster_labels, metric='euclidean')
    
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []  # cluster labels
    
    #plt.figure(figsize=(8, 6))
    for i in unique_clusters:
        # get and sort silhouette values for current cluster
        c_silhouette_vals = silhouette_vals[cluster_labels == i]
        c_silhouette_vals.sort()
        # updating y-axis bounds:
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals, 
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
    
    # adding average line and formatting
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, unique_clusters)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.title(f'Silhouette Plot {algorithm_name} for {dataset_name}')
    plt.tight_layout()
    plt.show()

def clustering_analysis(dataset_name):
    # loading data
    X, y = load_dataset(dataset_name)
    # standardizing the features
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    print(f"\n=== {dataset_name.upper()} DATASET CLUSTERING ANALYSIS ===")
    
     # Elbow Method and Silhouette Analysis
    print("\n--- Elbow Method & Silhouette Score Analysis ---")
    elbow_method(dataset_name, X_std)  

    # K-means clustering
    print("\n--- K-means Clustering ---")
    y_km = KMeans_clustering(dataset_name, X_std, y)
    plot_silhouette(X_std, y_km, dataset_name, 'K-means')
    
    # SciPy hierarchical clustering
    print("\n--- SciPy Hierarchical Clustering ---")
    y_scipy = Scipy_hierarchical_clustering(dataset_name, X_std, y)
    plot_silhouette(X_std, y_scipy, dataset_name, 'SciPy Hierarchical')
    
    # Scikit-learn hierarchical clustering
    print("\n--- Scikit-learn Hierarchical Clustering ---")
    y_sklearn = Sklearn_hierarchical_clustering(dataset_name, X_std, y)
    plot_silhouette(X_std, y_sklearn, dataset_name, 'Scikit-learn Hierarchical')

# Run analysis for both datasets
clustering_analysis('iris')
clustering_analysis('mnist')