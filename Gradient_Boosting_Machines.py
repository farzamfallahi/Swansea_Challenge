import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QAbstractItemView,
                             QSlider, QLineEdit, QCheckBox)
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class GBMParamsDialog(QDialog):
    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        self.setWindowTitle("Gradient Boosting Machine Parameters")
        self.columns = columns or []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Number of estimators
        self.n_estimators_slider, self.n_estimators_display = self.create_param_control(
            "Number of Estimators", 10, 1000, 100, 10, layout)

        # Learning rate
        self.learning_rate_slider, self.learning_rate_display = self.create_param_control(
            "Learning Rate", 0.01, 1.0, 0.1, 0.01, layout, is_float=True)

        # Max depth
        self.max_depth_slider, self.max_depth_display = self.create_param_control(
            "Max Depth", 1, 10, 3, 1, layout)

        # Test Size
        self.test_size_slider, self.test_size_display = self.create_param_control(
            "Test Size", 0.1, 0.5, 0.2, 0.1, layout, is_float=True)

        # Show Plots
        self.show_plots_checkbox = QCheckBox("Show Plots")
        self.show_plots_checkbox.setChecked(True)  # Default value
        layout.addWidget(self.show_plots_checkbox)

        # Add column selection for training
        train_layout = QVBoxLayout()
        train_layout.addWidget(QLabel("Select Training Columns:"))
        self.train_list = QListWidget()
        self.train_list.addItems(self.columns)
        self.train_list.setSelectionMode(QAbstractItemView.MultiSelection)
        train_layout.addWidget(self.train_list)
        layout.addLayout(train_layout)

        # Add column selection for testing
        test_layout = QVBoxLayout()
        test_layout.addWidget(QLabel("Select Target Column:"))
        self.test_list = QListWidget()
        self.test_list.addItems(self.columns)
        self.test_list.setSelectionMode(QAbstractItemView.SingleSelection)
        test_layout.addWidget(self.test_list)
        layout.addLayout(test_layout)

        # OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_param_control(self, label, min_val, max_val, default, step, parent_layout, is_float=False):
        param_layout = QVBoxLayout()
        param_label = QLabel(f"{label} ({min_val}-{max_val}):")
        param_layout.addWidget(param_label)

        input_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        
        if is_float:
            slider.setRange(int(min_val / step), int(max_val / step))
            slider.setValue(int(default / step))
        else:
            slider.setRange(min_val, max_val)
            slider.setValue(default)
        
        slider.setTickInterval(int((max_val - min_val) / 10))
        slider.setSingleStep(1)
        slider.setTickPosition(QSlider.TicksBelow)
        
        display = QLineEdit()
        display.setFixedWidth(60)
        display.setText(str(default))
        
        input_layout.addWidget(slider)
        input_layout.addWidget(display)
        
        param_layout.addLayout(input_layout)
        parent_layout.addLayout(param_layout)

        # Connect slider and line edit
        if is_float:
            slider.valueChanged.connect(lambda value: self.update_display(display, value * step))
        else:
            slider.valueChanged.connect(lambda value: self.update_display(display, value))
        display.editingFinished.connect(lambda: self.update_slider(slider, display, min_val, max_val, step, is_float))

        return slider, display

    def update_display(self, display, value):
        display.setText(f"{value:.2f}" if isinstance(value, float) else str(value))

    def update_slider(self, slider, display, min_val, max_val, step, is_float):
        try:
            value = float(display.text()) if is_float else int(display.text())
            if min_val <= value <= max_val:
                if is_float:
                    slider.setValue(int(value / step))
                else:
                    slider.setValue(value)
            else:
                self.update_display(display, slider.value() * step if is_float else slider.value())
        except ValueError:
            self.update_display(display, slider.value() * step if is_float else slider.value())

    def get_params(self):
        return {
            "n_estimators": self.n_estimators_slider.value(),
            "learning_rate": self.learning_rate_slider.value() * 0.01,
            "max_depth": self.max_depth_slider.value(),
            "test_size": self.test_size_slider.value() * 0.1,
            "show_plots": self.show_plots_checkbox.isChecked(),
            "train_columns": [item.text() for item in self.train_list.selectedItems()],
            "test_column": self.test_list.selectedItems()[0].text() if self.test_list.selectedItems() else None
        }

def prepare_and_train_gbm(data, params):
    # Extract parameters
    test_size = params['test_size']
    n_estimators = params['n_estimators']
    learning_rate = params['learning_rate']
    max_depth = params['max_depth']
    show_plots = params['show_plots']
    train_columns = params['train_columns']
    test_column = params['test_column']

    if not train_columns or not test_column:
        raise ValueError("Both train_columns and test_column must be specified.")

    # Separate features and target
    X = data[train_columns]
    y = data[test_column]

    # Determine if it's a classification or regression problem
    if y.dtype == 'object' or y.nunique() < 10:
        task = 'classification'
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
    else:
        task = 'regression'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create and train the GBM model
    if task == 'classification':
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

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
            from sklearn.metrics import confusion_matrix
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
        # Feature importance
        feature_names = numeric_features.tolist() + [f"{feature}_{cat}" for feature in categorical_features for cat in preprocessor.named_transformers_['cat'].categories_[list(categorical_features).index(feature)][1:]]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Plot training deviance
        test_score = np.zeros((n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_predict(X_test_processed)):
            if task == 'classification':
                test_score[i] = accuracy_score(y_test, y_pred)
            else:
                test_score[i] = mean_squared_error(y_test, y_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(n_estimators) + 1, model.train_score_, 'b-', label='Training Set Deviance')
        plt.plot(np.arange(n_estimators) + 1, test_score, 'r-', label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        plt.title('Deviance of GBM')
        plt.show()

    return model, X_train, X_test, y_train, y_test, preprocessor

# Example usage:
# dialog = GBMParamsDialog(columns=data.columns)
# if dialog.exec_():
#     params = dialog.get_params()
#     model, X_train, X_test, y_train, y_test, preprocessor = prepare_and_train_gbm(data, params)