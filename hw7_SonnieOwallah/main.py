import numpy as np
import pandas as pd
import time
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# loading Datasets
def load_dataset(name):
    if name == 'digits':
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target 
        print(f"Digit Data shape: {X.shape}")
        print(f"Digits Labels shape: {y.shape}")
    
    elif name == 'mammographic':
        # Try fetching with ucimlrepo first
        try:
            # fetch dataset 
            mammographic_mass = fetch_ucirepo(id=161) 
            
            # data (as pandas dataframes) 
            X = mammographic_mass.data.features 
            y = mammographic_mass.data.targets.values.ravel()   
            
            # variable information 
            print(mammographic_mass.variables) 
            # Print dataset info
            print("Dataset ID:", mammographic_mass.metadata["uci_id"])
            print("Dataset Name:", mammographic_mass.metadata["name"])
            print("\nFeature names:", mammographic_mass.variables["name"])
            print("\nOriginal shape (X):", X.shape)
            print("Original shape (y):", y.shape)

            # Check first 5 rows
            print("\nFirst 5 rows of features:\n", X[:5])
            print("\nFirst 5 target values:\n", y[:5])

            # Check for missing values
            print("\nMissing values in features:\n", pd.DataFrame(X).isna().sum())
            print("Missing values in target:\n", pd.DataFrame(y).isna().sum())
            
            # Impute missing values with median
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("Missing values after imputation:", np.isnan(X).sum())
        except ConnectionError:
            print("Error connecting to UCI server. Try loading from local file ")
            try: 
                mmass = pd.read_csv('mammographic_mass_data.csv', header=0)
                print(" mmass dataset shape:", mmass.shape)
                print("First 5 rows:\n", mmass.head())

                # separating features and target
                #X = mmass[['BI_RADS', 'Age', 'Shape', 'Margin', 'Mass_Density']].values
                features_df = mmass[['BI_RADS', 'Age', 'Shape', 'Margin', 'Mass_Density']].copy()
                y = mmass['Severity'].values

                # imputing missing values in numeric columns
                numeric_cols = ['BI_RADS', 'Age']
                imputer_numeric = SimpleImputer(strategy='median')
                features_df[numeric_cols] = imputer_numeric.fit_transform(features_df[numeric_cols])

                # imputing missing values in categorical columns with the most frequent value
                categorical_cols = ['Shape', 'Margin', 'Mass_Density']
                imputer_categorical = SimpleImputer(strategy='most_frequent')
                features_df[categorical_cols] = imputer_categorical.fit_transform(features_df[categorical_cols])

                # encoding categorical features
                features_encoded = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True) # drop_first to avoid multicollinearity

                X = features_encoded.values
                y = y.astype(int) # Ensure target is integer

                print("Features shape:", X.shape)
                print("Target shape:", y.shape)

            except FileNotFoundError:
                print("Mammographic dataset not found locally. Please download it and update the path.")
            
    else:
        raise ValueError("Dataset name invalid. Use 'digits' or 'mammographic'.")
        
    return X, y

# function to train and evaluate a single classifier
def train_evaluate_classifier(classifier, X_train, y_train, X_test, y_test, name):
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- {name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print("Classification Report:\n", report)

     # ROC curve plot (for binary classification only) 
    if len(np.unique(y_test)) == 2:  # binary check
    # skip ROC for hard voting classifiers
        if isinstance(classifier, VotingClassifier) and classifier.voting == 'hard':
            print("ROC skipped: Hard voting doesn't support probability estimates")
        else:
            try:
                if hasattr(classifier, "predict_proba"):
                    y_score = classifier.predict_proba(X_test)[:, 1]
                elif hasattr(classifier, "decision_function"):
                    y_score = classifier.decision_function(X_test)
                else:
                    raise AttributeError("No probability/decision function available")
                
            
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'{name} (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {name}')
                plt.legend(loc="lower right")
                plt.show()

            except Exception as e:
                print(f"ROC not available for {name}: {str(e)}")

    return accuracy, training_time, y_pred

## Main function ###
# running classifier and report performance
def main(dataset_name):
    X, y = load_dataset(dataset_name)

# splitting training and test for  datasets (80:20) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    print('Labels count in y:', np.bincount(y))
    print('Labels count in y_train:', np.bincount(y_train))
    print('Labels count in y_test:', np.bincount(y_test))

    #  standard schaler
    # sc = StandardScaler()
    # X_train_std = sc.fit_transform(X_train)
    # X_test_std = sc.transform(X_test)

    # defining base classifiers
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
    logistic = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', random_state=1, multi_class='ovr', max_iter=1000)
    knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

    # pipelines for scaling-dependent models
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', logistic)])
    pipe_knn = Pipeline([('scaler', StandardScaler()), ('knn', knn)])

    # base classifiers (for comparison with ensembles) 
    print("\n--- Base Classifiers ---")

    train_evaluate_classifier(tree, X_train, y_train, X_test, y_test, "Decision Tree")
    train_evaluate_classifier(pipe_lr, X_train, y_train, X_test, y_test, "Logistic Regression")
    train_evaluate_classifier(pipe_knn, X_train, y_train, X_test, y_test, "KNN")


    ### Ensemble Classifiers ###

    #  Random Forest
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=1, n_jobs=-1)
    train_evaluate_classifier(rf, X_train, y_train, X_test, y_test, "Random Forest")

    #  Bagging
    bag = BaggingClassifier(estimator=tree, n_estimators=100, random_state=1, n_jobs=-1)
    train_evaluate_classifier(bag, X_train, y_train, X_test, y_test, "Bagging")

    #  AdaBoost
    ada = AdaBoostClassifier(estimator=tree, n_estimators=100, learning_rate=0.1, random_state=1)
    train_evaluate_classifier(ada, X_train, y_train, X_test, y_test, "AdaBoost")

    ## Two other ensemble approaches##

    # Majority Vote Classifier
    mv_clf = VotingClassifier(estimators=[('lr', pipe_lr), ('dt', tree), ('knn', pipe_knn)], voting='hard')
    train_evaluate_classifier(mv_clf, X_train, y_train, X_test, y_test, "Majority Vote Classifier")

    #  Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
    train_evaluate_classifier(gb, X_train, y_train, X_test, y_test, "Gradient Boosting")


    # hyperparameter analysis

    print("\n--- Hyperparameter Tuning: AdaBoost (n_estimators) ---")
    for n in [50, 100, 150]:
        ada_tuned = AdaBoostClassifier(estimator=tree, n_estimators=n, learning_rate=0.1, random_state=1)
        train_evaluate_classifier(ada_tuned, X_train, y_train, X_test, y_test, f"AdaBoost (n_estimators={n})")

    print("\n--- Hyperparameter Tuning: Gradient Boosting (max_depth) ---")
    for depth in [1, 3, 5]:
        gb_tuned = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=1)
        train_evaluate_classifier(gb_tuned, X_train, y_train, X_test, y_test, f"Gradient Boosting (max_depth={depth})")


if __name__ == "__main__":
    print("\n--- Evaluating on Digits Dataset ---")
    main('digits')
    print("\n--- Evaluating on Mammographic Dataset ---")
    main('mammographic')