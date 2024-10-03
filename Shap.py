import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def perform_shap_analysis(model, X, feature_names, class_names=None, model_output="raw", num_display=20):
    """
    Perform SHAP analysis on a trained model.

    Parameters:
    - model: trained model object
    - X: input data (numpy array or pandas DataFrame)
    - feature_names: list of feature names
    - class_names: list of class names (for classification tasks)
    - model_output: "raw" for regression, "probability" for classification
    - num_display: number of features to display in summary plots

    Returns:
    - shap_values: SHAP values for the input data
    - explainer: SHAP explainer object
    """

    # Convert to pandas DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    # Create a SHAP explainer
    if model_output == "probability":
        explainer = shap.TreeExplainer(model, feature_names=feature_names, model_output="probability")
    else:
        explainer = shap.TreeExplainer(model, feature_names=feature_names)

    # Calculate SHAP values
    shap_values = explainer(X)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=num_display)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=num_display)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

    # SHAP dependence plots for top features
    top_features = shap_values.abs.mean(0).argsort()[-num_display:]
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values.values, X, feature_names=feature_names)
        plt.title(f"SHAP Dependence Plot for {feature_names[feature]}")
        plt.tight_layout()
        plt.show()

    # SHAP force plot for a single prediction (first instance)
    plt.figure(figsize=(20, 3))
    shap.force_plot(explainer.expected_value[0], shap_values.values[0], X.iloc[0], feature_names=feature_names, matplotlib=True)
    plt.title("SHAP Force Plot for Single Prediction")
    plt.tight_layout()
    plt.show()

    return shap_values, explainer

def shap_interaction_analysis(model, X, feature_names, num_interactions=10):
    """
    Perform SHAP interaction analysis on a trained model.

    Parameters:
    - model: trained model object
    - X: input data (numpy array or pandas DataFrame)
    - feature_names: list of feature names
    - num_interactions: number of top interactions to display

    Returns:
    - shap_interaction_values: SHAP interaction values
    """

    # Convert to pandas DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model, feature_names=feature_names)

    # Calculate SHAP interaction values
    shap_interaction_values = explainer.shap_interaction_values(X)

    # Sum of absolute SHAP interaction values over all instances
    total_interactions = np.abs(shap_interaction_values).sum(axis=0)

    # Get the indices of top interactions
    top_interactions = np.argsort(-total_interactions.sum(axis=1))[:num_interactions]

    # Plot top interactions
    plt.figure(figsize=(10, 8))
    for i, idx in enumerate(top_interactions):
        plt.bar(range(len(feature_names)), total_interactions.sum(axis=1)[idx], 
                bottom=np.sum([total_interactions.sum(axis=1)[top_interactions[j]] for j in range(i)], axis=0),
                label=feature_names[idx])

    plt.title("SHAP Feature Interactions")
    plt.xlabel("Features")
    plt.ylabel("Total Interaction SHAP Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return shap_interaction_values

# Example usage:
# Assuming you have a trained model, input data, and feature names
# shap_values, explainer = perform_shap_analysis(model, X, feature_names, model_output="raw")
# shap_interaction_values = shap_interaction_analysis(model, X, feature_names)