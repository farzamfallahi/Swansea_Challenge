# time_series_lime_svm.py
import numpy as np
import pandas as pd
from lime import lime_base
from lime.lime_tabular import LimeTabularExplainer
from LIME_For_Time import LimeTimeSeriesExplainer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]  # Return probabilities for the positive class
        else:
            return self.model.predict(X)

def show_time_series_lime_results_svm(model, X_train, instance_index, feature_names):
    if not hasattr(show_time_series_lime_results_svm, 'lime_window'):
        show_time_series_lime_results_svm.lime_window = QDialog()
        show_time_series_lime_results_svm.lime_layout = QVBoxLayout()
        show_time_series_lime_results_svm.lime_window.setLayout(show_time_series_lime_results_svm.lime_layout)
        show_time_series_lime_results_svm.lime_window.setWindowTitle("LIME Results - SVM")

    # Convert DataFrame to numpy array if it's not already
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
    else:
        X_train_array = X_train

    # Reshape the data
    X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], -1)

    explainer = LimeTimeSeriesExplainer(
        training_data=X_train_reshaped,
        feature_names=feature_names,
        class_names=None,
        feature_selection='auto',
        num_features=10,
        num_samples=5000,
        random_state=42
    )

    instance = X_train_array[instance_index]

    # Wrap the model in our ModelWrapper class
    model_wrapper = ModelWrapper(model)

    explanation = explainer.explain_instance(
        instance=instance,
        model=model_wrapper,
        labels=(1,),
        num_features=10,
        num_samples=5000,
        distance_metric='euclidean',
        model_regressor=None
    )

    # Clear previous contents
    for i in reversed(range(show_time_series_lime_results_svm.lime_layout.count())):
        show_time_series_lime_results_svm.lime_layout.itemAt(i).widget().setParent(None)

    # Display the explanation
    lime_text = QTextEdit()
    lime_text.setReadOnly(True)
    lime_text.setPlainText(str(explanation.as_list()))
    show_time_series_lime_results_svm.lime_layout.addWidget(lime_text)
    show_time_series_lime_results_svm.lime_window.show()