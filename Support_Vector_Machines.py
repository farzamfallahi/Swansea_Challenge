import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.svm import SVC, SVR

def prepare_and_train_svm(X, y, params):
    # Extract parameters
    kernel = params.get('kernel', 'rbf')
    C = params.get('C', 1.0)
    gamma = params.get('gamma', 'scale')
    degree = params.get('degree', 3)
    test_size = params.get('test_size', 0.2)
    show_plots = params.get('show_plots', True)

    # Handle NaN values in the target variable
    nan_mask = y.notnull()
    X = X[nan_mask]
    y = y[nan_mask]

    # Check if there are any samples left after removing NaN values
    if len(y) == 0:
        print("Error: All target values are NaN. Unable to train the model.")
        return None, None, None, None, None, None

    # Determine if it's a classification or regression problem
    if y.dtype == 'object' or y.nunique() < 10:
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Error: The target variable contains only one class: {unique_classes[0]}.")
            print("SVM requires at least two classes for classification.")
            print("Consider using regression if appropriate, or check your data.")
            return None, None, None, None, None, None
        task = 'classification'
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
    else:
        task = 'regression'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task == 'classification' else None)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if task == 'classification':
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42, probability=True)
        model.fit(X_train_processed, y_train)
    else:
        model = SVR(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Evaluate the model
    if task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        if show_plots:
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")

        if show_plots:
            # Plot predicted vs actual values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.show()

    if show_plots:
        # Feature importance (using permutation importance)
        perm_importance = permutation_importance(model, X_test_processed, y_test, n_repeats=10, random_state=42)
        feature_names = preprocessor.get_feature_names_out()
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    return model, X_train, X_test, y_train, y_test, preprocessor

def perform_svm_shap_analysis(model, X_test, feature_names):
    # Ensure X_test is 2D numpy array
    X_test = np.array(X_test)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    # Create a SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, 100))

    shap_values = explainer.shap_values(X_test)

    # Handle multi-class case
    if isinstance(shap_values, list):
        shap_values = np.abs(np.array(shap_values)).mean(axis=0)
    
    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Create the plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance for SVM')
    plt.tight_layout()
    
    return shap_values, plt.gcf()

def perform_svm_lime_analysis(model, X_train, X_test, feature_names, class_names=None):
    # Determine the mode based on the model type
    mode = 'classification' if isinstance(model, SVC) else 'regression'

    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode
    )
    
    # Explain an instance
    if mode == 'classification':
        exp = explainer.explain_instance(
            X_test[0],
            model.predict_proba,
            num_features=len(feature_names)
        )
    else:
        exp = explainer.explain_instance(
            X_test[0],
            model.predict,
            num_features=len(feature_names)
        )
    
    return exp 