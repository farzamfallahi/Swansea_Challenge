import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class KerasWrapper:
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        pass  # The model is already trained
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        if y.ndim == 1 or y.shape[1] == 1:  # Regression task
            return r2_score(y, y_pred)
        else:  # Classification task
            return accuracy_score(y.argmax(axis=1), y_pred.argmax(axis=1))

def get_eli5_neural_network_explanation(model, X, y, feature_names=None, n_iter=10):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
   
    # Wrap the Keras model
    wrapped_model = KerasWrapper(model)
    # Use permutation importance for neural networks
    perm_importance = permutation_importance(wrapped_model, X, y, n_repeats=n_iter, random_state=42)
   
    sorted_idx = perm_importance.importances_mean.argsort()
   
    explanation = "Neural Network Feature Importances (using Permutation Importance):\n\n"
    for idx in sorted_idx[::-1]:
        explanation += f"{feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} ± {perm_importance.importances_std[idx]:.4f}\n"
   
    return explanation

def get_eli5_explanation(model, X, y=None, mode='weights', feature_names=None):
    if hasattr(model, 'feature_importances_'):
        return get_eli5_text_explanation(model, X, feature_names)
    elif hasattr(model, 'coef_'):
        return get_eli5_text_explanation(model, X, feature_names)
    elif isinstance(model, tf.keras.Model):  # Check if it's a Keras model
        return get_eli5_neural_network_explanation(model, X, y, feature_names)
    elif isinstance(model, SVC):  # Check if it's an SVM model
        return get_eli5_svm_explanation(model, X, y, feature_names)
    elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):  # Check if it's a KNN model
        return get_permutation_importance_explanation(model, X, y, feature_names)
    else:
        return f"Model type {type(model)} not supported for ELI5 explanation."

def get_eli5_text_explanation(model, X, feature_names=None):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
   
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return "Model does not have feature importances or coefficients."

    sorted_idx = importances.argsort()
    explanation = "Feature Importances:\n"
    for idx in sorted_idx[::-1]:
        explanation += f"{feature_names[idx]}: {importances[idx]:.4f}\n"

    return explanation

def get_permutation_importance_explanation(model, X, y, feature_names=None, n_iter=10):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
   
    # Use permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=n_iter, random_state=42)
   
    sorted_idx = perm_importance.importances_mean.argsort()
   
    explanation = "Feature Importances (using Permutation Importance):\n\n"
    for idx in sorted_idx[::-1]:
        explanation += f"{feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} ± {perm_importance.importances_std[idx]:.4f}\n"
   
    return explanation

def get_eli5_svm_explanation(model, X, y, feature_names=None):
    return get_permutation_importance_explanation(model, X, y, feature_names)
