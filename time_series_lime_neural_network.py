# time_series_lime_neural_network.py

from lime import lime_base
from lime.lime_tabular import LimeTabularExplainer
from LIME_For_Time import LimeTimeSeriesExplainer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit

def show_time_series_lime_results_neural_network(model, X_train, instance_index, feature_names):
    if not hasattr(show_time_series_lime_results_neural_network, 'lime_window'):
        show_time_series_lime_results_neural_network.lime_window = QDialog()
        show_time_series_lime_results_neural_network.lime_layout = QVBoxLayout()
        show_time_series_lime_results_neural_network.lime_window.setLayout(show_time_series_lime_results_neural_network.lime_layout)
        show_time_series_lime_results_neural_network.lime_window.setWindowTitle("LIME Results - Neural Network")

    explainer = LimeTimeSeriesExplainer(
        training_data=X_train.reshape(X_train.shape[0], -1),
        feature_names=feature_names,
        class_names=None,
        feature_selection='auto',
        num_features=10,
        num_samples=5000,
        random_state=42
    )

    instance = X_train[instance_index]

    explanation = explainer.explain_instance(
        instance=instance,
        model=model,
        labels=(1,),
        num_features=10,
        num_samples=5000,
        distance_metric='euclidean',
        model_regressor=None
    )

    # Clear previous contents
    for i in reversed(range(show_time_series_lime_results_neural_network.lime_layout.count())):
        show_time_series_lime_results_neural_network.lime_layout.itemAt(i).widget().setParent(None)

    # Display the explanation
    lime_text = QTextEdit()
    lime_text.setReadOnly(True)
    lime_text.setPlainText(str(explanation.as_list()))
    show_time_series_lime_results_neural_network.lime_layout.addWidget(lime_text)

    show_time_series_lime_results_neural_network.lime_window.show()