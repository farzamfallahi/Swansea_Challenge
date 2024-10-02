import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime.lime_tabular
from lime import lime_tabular
import lime
from sklearn.pipeline import make_pipeline

def perform_lime_analysis(model, X_train, X_test, feature_names, class_names=None, mode='classification', num_features=10, num_samples=5000):
    """
    Perform LIME analysis on a trained model.

    Parameters:
    - model: trained model object
    - X_train: training data (numpy array or pandas DataFrame)
    - X_test: test data (numpy array or pandas DataFrame)
    - feature_names: list of feature names
    - class_names: list of class names (for classification tasks)
    - mode: 'classification' or 'regression'
    - num_features: number of top features to show in the explanation
    - num_samples: number of samples to use for LIME explainer

    Returns:
    - explainer: LIME explainer object
    - exp: LIME explanation for a sample instance
    """

    # Convert to numpy arrays if pandas DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        mode=mode,
        feature_names=feature_names,
        class_names=class_names,
        verbose=True,
        random_state=42
    )

    # Choose a random instance to explain
    instance_idx = np.random.randint(0, X_test.shape[0])
    instance = X_test[instance_idx]

    # Generate explanation
    exp = explainer.explain_instance(
        instance, 
        model.predict_proba if mode == 'classification' else model.predict, 
        num_features=num_features,
        num_samples=num_samples
    )

    # Print the explanation
    print("LIME Explanation:")
    print(exp.as_list())

    # Visualize the explanation
    fig = plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title("LIME Explanation for Instance")
    plt.tight_layout()
    plt.show()

    # If it's a classification task, show probability prediction
    if mode == 'classification':
        print("\nPredicted probabilities:")
        print(dict(zip(class_names, model.predict_proba([instance])[0])))

    return explainer, exp

def lime_multiple_instances(model, X_train, X_test, feature_names, class_names=None, mode='classification', num_features=10, num_samples=5000, num_instances=5):
    """
    Perform LIME analysis on multiple instances.

    Parameters:
    - model: trained model object
    - X_train: training data (numpy array or pandas DataFrame)
    - X_test: test data (numpy array or pandas DataFrame)
    - feature_names: list of feature names
    - class_names: list of class names (for classification tasks)
    - mode: 'classification' or 'regression'
    - num_features: number of top features to show in the explanation
    - num_samples: number of samples to use for LIME explainer
    - num_instances: number of instances to explain

    Returns:
    - explainer: LIME explainer object
    - explanations: list of LIME explanations for multiple instances
    """

    # Convert to numpy arrays if pandas DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        mode=mode,
        feature_names=feature_names,
        class_names=class_names,
        verbose=True,
        random_state=42
    )

    explanations = []

    for i in range(num_instances):
        # Choose a random instance to explain
        instance_idx = np.random.randint(0, X_test.shape[0])
        instance = X_test[instance_idx]

        # Generate explanation
        exp = explainer.explain_instance(
            instance, 
            model.predict_proba if mode == 'classification' else model.predict, 
            num_features=num_features,
            num_samples=num_samples
        )

        explanations.append(exp)

        # Print the explanation
        print(f"\nLIME Explanation for Instance {i+1}:")
        print(exp.as_list())

        # Visualize the explanation
        fig = plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title(f"LIME Explanation for Instance {i+1}")
        plt.tight_layout()
        plt.show()

        # If it's a classification task, show probability prediction
        if mode == 'classification':
            print("\nPredicted probabilities:")
            print(dict(zip(class_names, model.predict_proba([instance])[0])))

    return explainer, explanations

# Example usage:
# Assuming you have a trained model, training data, test data, and feature names
# explainer, exp = perform_lime_analysis(model, X_train, X_test, feature_names, class_names, mode='classification')
# 
# To explain multiple instances:
# explainer, explanations = lime_multiple_instances(model, X_train, X_test, feature_names, class_names, mode='classification', num_instances=5)

def perform_gbm_lime_analysis(model, X_train, X_test, feature_names, class_names):
    # Convert to numpy array if it's a DataFrame
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )

    # Choose a random instance from X_test
    idx = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(
        X_test[idx], 
        model.predict_proba, 
        num_features=len(feature_names)
    )

    return explainer, exp
