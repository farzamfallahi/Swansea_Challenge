
import sys
import os
import scipy.sparse
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV  # Add this line
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from lime import lime_base
from lime.lime_tabular import LimeTabularExplainer
from LIME_For_Time import LimeTimeSeriesExplainer
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import seaborn as sns
from ELI5 import get_eli5_explanation
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QDialog, QMessageBox, QLabel, QTableWidget, QTableWidgetItem,
    QInputDialog, QTableView, QLineEdit, QTextEdit, QDialogButtonBox, QScrollArea,
    QGroupBox, QCheckBox, QRadioButton, QListWidget, QListWidgetItem, QTabWidget
)
from scipy import stats
import scipy
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QTextCursor
from sklearn.ensemble import IsolationForest
from Logistic_Regression import prepare_and_train_logistic_regression
from sklearn.inspection import permutation_importance
from logistic_regression_params import LogisticRegressionParamsDialog
from PyQt5.QtCore import Qt, QAbstractTableModel
from prepare_data_dialog import PrepareDataDialog
from data_viewer import DataViewerDialog
from data_description_dialog import DataDescriptionDialog
from neural_network_params import NeuralNetworkParamsDialog
from PyQt5.QtGui import QIntValidator
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from Gradient_Boosting_Machines import prepare_and_train_gbm
from gradient_boosting_params import GradientBoostingParamsDialog
from Neural_Network import prepare_and_train_neural_network
from Lime import perform_lime_analysis
from Lime import perform_gbm_lime_analysis
from Shap import perform_shap_analysis, shap_interaction_analysis
from preparation import handle_missing_values, normalize_features
from plotting_functions import plot_one_dimensional_histogram, plot_two_dimensional_histogram, plot_scatter
from decision_tree import show_decision_tree
from decision_tree_params import DecisionTreeParamsDialog
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import f1_score, mean_squared_error, r2_score, accuracy_score, classification_report
from Support_Vector_Machines import prepare_and_train_svm, perform_svm_shap_analysis, perform_svm_lime_analysis
from K_Nearest_Neighbors import prepare_and_train_knn
from support_vector_machines_params import SVMParamsDialog
from knn_params import KNNParamsDialog
from time_series_lime_neural_network import show_time_series_lime_results_neural_network
from time_series_lime_knn import show_time_series_lime_results_knn
from time_series_lime_gbm import show_time_series_lime_results_gbm
from time_series_lime_svm import show_time_series_lime_results_svm
from confusion_matrix import confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import lime
import lime.lime_tabular
import tensorflow as tf
Model = tf.keras.models.Model  # Add this line to import the Model class
from shap import DeepExplainer
from PyQt5.QtWidgets import (QSlider, QComboBox)
from PyQt5.QtCore import Qt


class NonModalDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)

    def exec_(self):
        self.show()

class MessageBox(NonModalDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        label = QLabel(message)
        layout.addWidget(label)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.close)
        layout.addWidget(ok_button)
        self.setLayout(layout)

# Replace QMessageBox.warning calls with:
# MessageBox("Warning", "Your warning message here", self).show()

class PrepareDataDialog(NonModalDialog):
    def __init__(self, data, file_name, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.original_data = data.copy()  # Keep a copy of the original data
        self.data = data.copy()  # Working copy of the data
        self.file_name = file_name
        self.merged_data = None  # Initialize merged_data as None
        self.setWindowTitle(f"Prepare Data from {file_name}")
        self.resize(1000, 600)

        # Main layout
        main_layout = QHBoxLayout(self)

        # Left side layout for the table
        left_layout = QVBoxLayout()

        self.table = QTableWidget()
        self.update_table()

        left_layout.addWidget(self.table)

        # Right side layout for data preparation options
        right_layout = QVBoxLayout()

        # Recommendation button
        self.recommendation_button = QPushButton("Get Preparation Recommendations")
        self.recommendation_button.clicked.connect(self.show_recommendations)
        right_layout.addWidget(self.recommendation_button)

        # Handle Missing Values
        self.missing_values_group = QGroupBox("Handle Missing Values")
        missing_values_layout = QVBoxLayout()
        imputation_methods = ['mean', 'median', 'mode', 'knn', 'multiple', 'regression', 'decision_tree', 'flag']
        for method in imputation_methods:
            button = QPushButton(f"{method.capitalize()} Imputation")
            button.clicked.connect(lambda checked, m=method: self.handle_imputation(m))
            missing_values_layout.addWidget(button)
        self.missing_values_group.setLayout(missing_values_layout)
        right_layout.addWidget(self.missing_values_group)

        # Handle Outliers
        self.outliers_group = QGroupBox("Handle Outliers")
        outliers_layout = QVBoxLayout()
        outlier_methods = ['zscore', 'iqr', 'isolation_forest']
        for method in outlier_methods:
            button = QPushButton(f"{method.capitalize()} Outlier Handling")
            button.clicked.connect(lambda checked, m=method: self.handle_outliers(m))
            outliers_layout.addWidget(button)
        self.outliers_group.setLayout(outliers_layout)
        right_layout.addWidget(self.outliers_group)

        # Drop Column/Row
        self.drop_button = QPushButton("Drop Column/Row")
        self.drop_button.clicked.connect(self.handle_drop_column_row)
        right_layout.addWidget(self.drop_button)

        # Add Column/Row
        self.add_button = QPushButton("Add Column/Row")
        self.add_button.clicked.connect(self.handle_add_column_row)
        right_layout.addWidget(self.add_button)

        # Normalization
        self.normalization_group = QGroupBox("Normalization")
        normalization_layout = QVBoxLayout()
        normalization_methods = ['min_max', 'z_score', 'robust', 'max_abs']
        for method in normalization_methods:
            button = QPushButton(f"{method.replace('_', ' ').capitalize()} Scaling")
            button.clicked.connect(lambda checked, m=method: self.handle_normalization_method(m))
            normalization_layout.addWidget(button)
        self.normalization_group.setLayout(normalization_layout)
        right_layout.addWidget(self.normalization_group)

        # Add Save button
        self.save_button = QPushButton("Save Prepared Data")
        self.save_button.clicked.connect(self.handle_save_click)
        right_layout.addWidget(self.save_button)

        # Add Undo button
        self.undo_button = QPushButton("Undo Last Change")
        self.undo_button.clicked.connect(self.undo_last_change)
        right_layout.addWidget(self.undo_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        # Initialize undo stack
        self.undo_stack = [self.data.copy()]

        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        right_layout.addWidget(self.apply_button)  # Add to right_layout instead of undefined 'layout'

    def apply_changes(self):
        # Apply the changes to the data
        self.main_window.data = self.get_prepared_data()
        self.main_window.update_table()
        self.close()

    def undo_last_change(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop()  # Remove the current state
            self.data = self.undo_stack[-1].copy()  # Set the data to the previous state
            self.update_table()
            QMessageBox.information(self, "Undo", "Last change has been undone.")
        else:
            QMessageBox.information(self, "Undo", "No more changes to undo.")

    def handle_save_click(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if file_path:
                try:
                    self.data.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Save Successful", f"Prepared data saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")

    def get_prepared_data(self):
        return self.data

    def handle_save_click(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if file_path:
                try:
                    self.data.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Save Successful", f"Prepared data saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")

    def show_recommendations(self):
        recommendations = self.get_recommendations()
        dialog = QDialog(self)
        dialog.setWindowTitle("Data Preparation Recommendations")
        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("<h3>Recommendations:</h3>" + "<br>".join(recommendations))
        layout.addWidget(text_edit)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def get_recommendations(self):
        recommendations = []

        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            recommendations.append("Handle missing values:")
            if self.data.select_dtypes(include=['number']).columns.any():
                recommendations.append("- Consider Mean or Median Imputation for numeric columns")
            if self.data.select_dtypes(include=['object']).columns.any():
                recommendations.append("- Consider Mode Imputation for categorical columns")
            recommendations.append("- KNN Imputation can be effective for datasets with correlations between features")
            recommendations.append("- Multiple Imputation or Regression Imputation for more sophisticated approaches")
            recommendations.append("- Decision Tree Imputation can capture non-linear relationships")
            recommendations.append("- Flag Imputation to keep track of which values were imputed")

        # Check for categorical variables
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            recommendations.append("Consider encoding categorical variables:")
            recommendations.append("- One-hot encoding for nominal categorical variables")
            recommendations.append("- Ordinal encoding for ordinal categorical variables")

        # Check for numeric variables
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        if not numeric_columns.empty:
            recommendations.append("Consider normalizing numeric features:")
            recommendations.append("- Min-Max Scaling for features with a bounded range")
            recommendations.append("- Z-Score Scaling for normally distributed features")
            recommendations.append("- Robust Scaling for data with outliers")
            recommendations.append("- Max Abs Scaling for sparse data")

        # Check for high cardinality categorical variables
        high_cardinality = [col for col in categorical_columns if self.data[col].nunique() > 10]
        if high_cardinality:
            recommendations.append("Consider dimensionality reduction for high cardinality categorical variables:")
            recommendations.append("- Grouping less frequent categories")
            recommendations.append("- Using embedding techniques")

        # Check for highly correlated features
        if not numeric_columns.empty:
            corr_matrix = self.data[numeric_columns].corr()
            high_corr = (corr_matrix.abs() > 0.8).sum().sum() > len(numeric_columns)
            if high_corr:
                recommendations.append("Consider feature selection or dimensionality reduction for highly correlated features")

        # General recommendations
        recommendations.append("Always split your data into training and testing sets before applying transformations")
        recommendations.append("Consider cross-validation for more robust model evaluation")

        return recommendations

    def handle_add_column_row(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Column/Row")
        layout = QVBoxLayout(dialog)

        self.add_column_radio = QRadioButton("Add Column")
        self.add_row_radio = QRadioButton("Add Row")
        self.add_column_radio.setChecked(True)
        layout.addWidget(self.add_column_radio)
        layout.addWidget(self.add_row_radio)

        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        values_layout = QHBoxLayout()
        values_label = QLabel("Values (comma-separated):")
        self.values_input = QLineEdit()
        values_layout.addWidget(values_label)
        values_layout.addWidget(self.values_input)
        layout.addLayout(values_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_():
            if self.add_column_radio.isChecked():
                self.add_column()
            else:
                self.add_row()

    def add_column(self):
        column_name = self.name_input.text()
        values = self.values_input.text().split(',')
        
        if len(values) != len(self.data):
            QMessageBox.warning(self, "Invalid Input", "Number of values must match the number of rows in the dataset.")
            return

        self.data[column_name] = values
        self.update_table()
        self.undo_stack.append(self.data.copy())
        QMessageBox.information(self, "Column Added", f"Added column: {column_name}")


    def add_column(self):
        column_name = self.name_input.text()
        values = self.values_input.text().split(',')
        
        if len(values) != len(self.data):
            QMessageBox.warning(self, "Invalid Input", "Number of values must match the number of rows in the dataset.")
            return

        self.data[column_name] = values
        self.update_table()
        QMessageBox.information(self, "Column Added", f"Added column: {column_name}")

    def add_row(self):
        try:
            row_index = int(self.name_input.text())
            values = self.values_input.text().split(',')
            
            if len(values) != len(self.data.columns):
                QMessageBox.warning(self, "Invalid Input", "Number of values must match the number of columns in the dataset.")
                return

            new_row = pd.DataFrame([values], columns=self.data.columns, index=[row_index])
            self.data = pd.concat([self.data.iloc[:row_index], new_row, self.data.iloc[row_index:]]).reset_index(drop=True)
            self.update_table()
            self.undo_stack.append(self.data.copy())
            QMessageBox.information(self, "Row Added", f"Added row at index: {row_index}")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for the row index.")

    def handle_drop_column_row(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Columns to Drop")
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget(dialog)
        for column in self.data.columns:
            item = QListWidgetItem(str(column))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_():
            columns_to_drop = [list_widget.item(i).text() for i in range(list_widget.count()) 
                               if list_widget.item(i).checkState() == Qt.Checked]
            if columns_to_drop:
                self.data = self.data.drop(columns=columns_to_drop)
                self.update_table()
                QMessageBox.information(self, "Drop Successful", f"Dropped columns: {', '.join(columns_to_drop)}")
            else:
                QMessageBox.information(self, "No Selection", "No columns were selected to drop.")

    def handle_normalization_method(self, method):
        self.data = normalize_features(self.data, method=method)
        self.update_table()
        self.undo_stack.append(self.data.copy())
        QMessageBox.information(self, "Normalization", f"{method.replace('_', ' ').capitalize()} scaling completed.")

    def update_table(self):
        self.table.setRowCount(self.data.shape[0])
        self.table.setColumnCount(self.data.shape[1])
        self.table.setHorizontalHeaderLabels(self.data.columns.astype(str))

        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                item = QTableWidgetItem(str(self.data.iat[row, col]))
                self.table.setItem(row, col, item)

        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

    def accept(self):
        # When OK is clicked, update the original data
        self.main_window.data = self.data
        super().accept()

    def handle_missing_values(self):
        self.missing_values_widget.setVisible(not self.missing_values_widget.isVisible())

    def handle_imputation(self, method):
        self.data = handle_missing_values(self.data, method=method)
        self.update_table()
        self.undo_stack.append(self.data.copy())
        QMessageBox.information(self, "Imputation", f"{method.capitalize()} imputation completed.")

    def handle_normalization(self):
        self.normalization_widget.setVisible(not self.normalization_widget.isVisible())

    def show_updated_dataset(self, updated_dataset):
        dialog = QDialog(self)
        dialog.setWindowTitle("Updated Dataset")
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setRowCount(updated_dataset.shape[0])
        table.setColumnCount(updated_dataset.shape[1])
        table.setHorizontalHeaderLabels(updated_dataset.columns.astype(str))

        for row in range(updated_dataset.shape[0]):
            for col in range(updated_dataset.shape[1]):
                item = QTableWidgetItem(str(updated_dataset.iat[row, col]))
                table.setItem(row, col, item)

        table.resizeRowsToContents()
        table.resizeColumnsToContents()

        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.show()  # Show the dialog non-modally

    def show_normalized_datasets(self, normalized_datasets):
        for dataset in normalized_datasets:
            dialog = QDialog(self)
            dialog.setWindowTitle("Normalized Dataset")
            dialog.resize(800, 600)
            layout = QVBoxLayout(dialog)

            table_view = QTableView()
            table_model = PandasModel(dataset)
            table_view.setModel(table_model)
            table_view.resizeRowsToContents()
            table_view.resizeColumnsToContents()

            layout.addWidget(table_view)
            dialog.setLayout(layout)
            dialog.show()

    def handle_merge_click(self):
        # Add your merge functionality here
        pass
    
    def handle_transpose_click(self):
        # Add your transpose functionality here
        pass

    def toggle_plotting_buttons(self):
        # Add 
        pass

    def handle_scatter_click(self):
        # Add your scatter plot functionality here
        pass

    def handle_hist1d_click(self):
        # Add your scatter plot functionality here
        pass

    def handle_hist2d_click(self):
        # Add your scatter plot functionality here
        pass

    def show_missing_values_dialog(self):
        # Add your scatter plot functionality here
        pass

    def handle_preparation_click(self):
        # Add your scatter plot functionality here
        pass

    def toggle_model_buttons(self):
        # Add your model button toggle functionality here
        pass

    def handle_decision_tree_click(self):
        # Add your model button toggle functionality here
        pass

    def handle_neural_network_click(self):
        # Add your model button toggle functionality here
        pass

    def handle_knn_click(self):
        # Add your model button toggle functionality here
        pass

    def handle_svm_click(self):
        # Add your model button toggle functionality here
        pass

    def handle_gbm_click(self):
        # Add your model button toggle functionality here
        pass

    def handle_logistic_regression_click(self):
        # Add your model button toggle functionality here
        pass

    def handle_drop_column_row(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Columns to Drop")
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget(dialog)
        for column in self.data.columns:
            item = QListWidgetItem(str(column))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_():
            columns_to_drop = [list_widget.item(i).text() for i in range(list_widget.count()) 
                               if list_widget.item(i).checkState() == Qt.Checked]
            if columns_to_drop:
                self.data = self.data.drop(columns=columns_to_drop)
                self.update_table()
                self.undo_stack.append(self.data.copy())
                QMessageBox.information(self, "Drop Successful", f"Dropped columns: {', '.join(columns_to_drop)}")
            else:
                QMessageBox.information(self, "No Selection", "No columns were selected to drop.")

    def handle_outliers(self, method):
        if method == 'zscore':
            self.handle_zscore_outliers()
        elif method == 'iqr':
            self.handle_iqr_outliers()
        elif method == 'isolation_forest':
            self.handle_isolation_forest_outliers()
        self.undo_stack.append(self.data.copy())

    def handle_zscore_outliers(self, threshold=3):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        z_scores = stats.zscore(self.data[numeric_columns])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)
        self.data = self.data[filtered_entries]
        self.update_table()
        QMessageBox.information(self, "Outlier Handling", f"Z-score outlier handling completed. Removed {self.data.shape[0] - filtered_entries.sum()} rows.")

    def handle_iqr_outliers(self, factor=1.5):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        Q1 = self.data[numeric_columns].quantile(0.25)
        Q3 = self.data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        filtered_entries = ~((self.data[numeric_columns] < (Q1 - factor * IQR)) | (self.data[numeric_columns] > (Q3 + factor * IQR))).any(axis=1)
        self.data = self.data[filtered_entries]
        self.update_table()
        QMessageBox.information(self, "Outlier Handling", f"IQR outlier handling completed. Removed {self.data.shape[0] - filtered_entries.sum()} rows.")

    def handle_isolation_forest_outliers(self, contamination=0.1):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.data[numeric_columns])
        self.data = self.data[outlier_labels != -1]
        self.update_table()
        QMessageBox.information(self, "Outlier Handling", f"Isolation Forest outlier handling completed. Removed {(outlier_labels == -1).sum()} rows.")

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class MissingValuesDialog(QDialog):
    def __init__(self, missing_values, min_values, max_values, total_rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Missing Values, Min and Max Percentage")
        self.resize(600, 400)  # Set initial size

        main_layout = QVBoxLayout()

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(scroll_area)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        self.label = QLabel()
        self.label.setWordWrap(True)  # Enable word wrapping
        scroll_layout.addWidget(self.label)

        main_layout.addWidget(scroll_area)

        self.set_missing_values(missing_values, min_values, max_values, total_rows)

        self.setLayout(main_layout)


    def set_missing_values(self, missing_values, min_values, max_values, total_rows):
        text = "Missing Values, Min and Max Percentage:\n"
        for column in missing_values.index:
            text += f"{column}:\n"
            text += f"  Missing: {missing_values[column]} ({missing_values[column] / total_rows * 100:.2f}%)\n"
            text += f"  Min: {min_values[column]}\n"
            text += f"  Max: {max_values[column]}\n"
        self.label.setText(text)

class DataWindow(NonModalDialog):
    def __init__(self, data, file_name, main_window, parent=None, merged_data=None):
        super().__init__(parent)
        self.main_window = main_window
        self.data = data
        self.file_name = file_name
        self.merged_data = merged_data
        self.dialog_history = []  # Add this line
        # Set the window title with the number of rows and columns
        num_rows, num_columns = self.data.shape
        self.setWindowTitle(f"Data from {file_name} - Rows: {num_rows}, Columns: {num_columns}")
        self.resize(800, 600)
        self.decision_tree_dialog = None  # Add this line to store the dialog reference


        # Main layout
        main_layout = QHBoxLayout(self)

        # Left side layout for the table and other existing buttons
        left_layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns.astype(str))
        self.table.setSelectionBehavior(QTableWidget.SelectItems)
        self.table.setSelectionMode(QTableWidget.MultiSelection)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.edit_column_title)
        self.table.itemSelectionChanged.connect(self.update_merge_button_state)

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[row, col]))
                self.table.setItem(row, col, item)

        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

        left_layout.addWidget(self.table)

        # Add other existing buttons to left_layout
        self.merge_button = QPushButton("Merge/Concatenate")
        self.merge_button.clicked.connect(self.handle_merge_click)
        self.merge_button.setEnabled(True)
        left_layout.addWidget(self.merge_button)

        self.transpose_button = QPushButton("Transpose")
        self.transpose_button.clicked.connect(self.handle_transpose_click)
        left_layout.addWidget(self.transpose_button)

        self.plotting_button = QPushButton("Plotting")
        self.plotting_button.clicked.connect(self.toggle_plotting_buttons)
        left_layout.addWidget(self.plotting_button)

        self.scatter_button = QPushButton("Scatter")
        self.scatter_button.clicked.connect(self.handle_scatter_click)
        self.scatter_button.setVisible(False)
        left_layout.addWidget(self.scatter_button)

        self.hist1d_button = QPushButton("Histogram (1D)")
        self.hist1d_button.clicked.connect(self.handle_hist1d_click)
        self.hist1d_button.setVisible(False)
        left_layout.addWidget(self.hist1d_button)

        self.hist2d_button = QPushButton("Histogram (2D)")
        self.hist2d_button.clicked.connect(self.handle_hist2d_click)
        self.hist2d_button.setVisible(False)
        left_layout.addWidget(self.hist2d_button)

        self.missing_values_button = QPushButton("Missing Values, Min and Max Percentage")
        self.missing_values_button.clicked.connect(self.show_missing_values_dialog)
        left_layout.addWidget(self.missing_values_button)

        self.recommendation_button = QPushButton("Recommendation")
        self.recommendation_button.clicked.connect(self.show_recommendation)
        left_layout.addWidget(self.recommendation_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.handle_save_click)
        left_layout.addWidget(self.save_button)

        main_layout.addLayout(left_layout)

        # Right side layout for the Models button and other model buttons
        right_layout = QVBoxLayout()
        self.models_button = QPushButton("Models")
        self.models_button.clicked.connect(self.toggle_model_buttons)
        right_layout.addWidget(self.models_button)
        right_layout.addStretch()  # This pushes the Models button to the top

        # Create model buttons (initially hidden)
        self.preparation_button = QPushButton("Preparation")
        self.preparation_button.clicked.connect(self.handle_preparation_click)
        self.preparation_button.hide()
        right_layout.addWidget(self.preparation_button)

        # Create model buttons (initially hidden)
        self.decision_tree_button = QPushButton("Decision Tree")
        self.decision_tree_button.clicked.connect(self.handle_decision_tree_click)
        self.decision_tree_button.hide()
        right_layout.addWidget(self.decision_tree_button)

        self.neural_network_button = QPushButton("Neural Network")
        self.neural_network_button.clicked.connect(self.handle_neural_network_click)
        self.neural_network_button.hide()
        right_layout.addWidget(self.neural_network_button)

        self.knn_button = QPushButton("K-Nearest Neighbors")
        self.knn_button.clicked.connect(self.handle_knn_click)
        self.knn_button.hide()
        right_layout.addWidget(self.knn_button)

        self.svm_button = QPushButton("Support Vector Machines")
        self.svm_button.clicked.connect(self.handle_svm_click)
        self.svm_button.hide()
        right_layout.addWidget(self.svm_button)

        self.gbm_button = QPushButton("Gradient Boosting Machines")
        self.gbm_button.clicked.connect(self.handle_gbm_click)
        self.gbm_button.hide()
        right_layout.addWidget(self.gbm_button)

        self.logistic_regression_button = QPushButton("Logistic Regression")
        self.logistic_regression_button.clicked.connect(self.handle_logistic_regression_click)
        self.logistic_regression_button.hide()
        right_layout.addWidget(self.logistic_regression_button)

        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def toggle_model_buttons(self):
        self.preparation_button.setVisible(not self.preparation_button.isVisible())
        self.decision_tree_button.setVisible(not self.decision_tree_button.isVisible())
        self.neural_network_button.setVisible(not self.neural_network_button.isVisible())
        self.knn_button.setVisible(not self.knn_button.isVisible())
        self.svm_button.setVisible(not self.svm_button.isVisible())
        self.gbm_button.setVisible(not self.gbm_button.isVisible())
        self.logistic_regression_button.setVisible(not self.logistic_regression_button.isVisible())

    def toggle_plotting_buttons(self):
        self.scatter_button.setVisible(not self.scatter_button.isVisible())
        self.hist1d_button.setVisible(not self.hist1d_button.isVisible())
        self.hist2d_button.setVisible(not self.hist2d_button.isVisible())

    def edit_column_title(self, index):
        dialog = ColumnTitleDialog(str(self.data.columns[index]), self.data.shape[1], self)
        if dialog.exec_():
            new_title = dialog.title_edit.text()
            self.data = self.data.rename(columns={self.data.columns[index]: new_title})
            self.table.setHorizontalHeaderItem(index, QTableWidgetItem(new_title))

            # Handle column reordering
            if dialog.order_edit.text():
                try:
                    new_order = int(dialog.order_edit.text())
                    if 0 <= new_order < self.data.shape[1]:
                        cols = list(self.data.columns)
                        cols.insert(new_order, cols.pop(index))
                        self.data = self.data[cols]
                        self.update_table()
                    else:
                        QMessageBox.warning(self, "Invalid Order", "Column order must be between 0 and {}".format(self.data.shape[1] - 1))
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for the column order.")    
    
    def update_merge_button_state(self):
        selected_items = self.table.selectedItems()
        self.merge_button.setEnabled(bool(selected_items))

    def handle_merge_click(self):
        merge_dialog = MultiDatasetMergeDialog(self.main_window.data_windows, self.main_window, self)
        merge_dialog.show()
        if merge_dialog.exec_():  # Use exec_() for modal behavior
            merged_data = merge_dialog.get_merged_data()
            if merged_data is not None:
                new_window = DataWindow(merged_data, "Merged from multiple files", self.main_window, self)
                self.main_window.data_windows.append(new_window)
                new_window.show()
            else:
                QMessageBox.warning(self, "Merge Error", "No columns were selected for merging.")
        self.table.clearSelection()

    def handle_transpose_click(self):
        self.data = self.data.transpose()
        self.table.setRowCount(self.data.shape[0])
        self.table.setColumnCount(self.data.shape[1])
        self.table.setHorizontalHeaderLabels(self.data.columns.astype(str))
        self.table.setVerticalHeaderLabels(self.data.index.astype(str))

        # Clear existing items
        self.table.clearContents()

        # Set items with data values
        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                item = QTableWidgetItem(str(self.data.iat[row, col]))
                self.table.setItem(row, col, item)

        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        
        # Update the window title with the new number of rows and columns
        num_rows, num_columns = self.data.shape
        self.setWindowTitle(f"Data from {self.file_name} - Rows: {num_rows}, Columns: {num_columns}")

    def get_selected_data(self):
        selected_rows = set(item.row() for item in self.table.selectedItems())
        selected_cols = set(item.column() for item in self.table.selectedItems())
        selected_data = self.data.iloc[list(selected_rows), list(selected_cols)]
        return selected_data

    def handle_scatter_click(self):
        selected_items = self.table.selectedItems()
        if len(selected_items) >= 2:
            col_indices = list(set(item.column() for item in selected_items))
            if len(col_indices) == 2:
                x = self.data.iloc[:, col_indices[0]]
                y = self.data.iloc[:, col_indices[1]]
                plot_scatter(x, y, title='Scatter Plot', xlabel=x.name, ylabel=y.name)
            else:
                QMessageBox.warning(self, "Selection Error", "Please select exactly two columns for a scatter plot.")

    def handle_hist1d_click(self):
        selected_items = self.table.selectedItems()
        if selected_items:
            col_indices = list(set(item.column() for item in selected_items))
            if len(col_indices) == 1:
                data = self.data.iloc[:, col_indices[0]]
                plot_one_dimensional_histogram(data, title='One-Dimensional Histogram', xlabel=data.name, ylabel='Frequency')
            else:
                QMessageBox.warning(self, "Selection Error", "Please select exactly one column for a 1D histogram.")

    def handle_hist2d_click(self):
        selected_items = self.table.selectedItems()
        if len(selected_items) >= 2:
            col_indices = list(set(item.column() for item in selected_items))
            if len(col_indices) == 2:
                x = self.data.iloc[:, col_indices[0]]
                y = self.data.iloc[:, col_indices[1]]
                try:
                    plot_two_dimensional_histogram(x, y, title='Two-Dimensional Histogram', xlabel=x.name, ylabel=y.name)
                except Exception as e:
                    error_message = f"Failed to create 2D histogram: {str(e)}\n\n"
                    error_message += "This could be due to incompatible data types or other plotting issues."
                    QMessageBox.warning(self, "Plotting Error", error_message)
            else:
                QMessageBox.warning(self, "Selection Error", "Please select exactly two columns for a 2D histogram.")
        else:
            QMessageBox.warning(self, "Selection Error", "Please select at least two columns for a 2D histogram.")

    def show_missing_values_dialog(self):
        missing_values = self.data.isnull().sum()
        total_rows = len(self.data)
        
        # Calculate minimum and maximum values for each column
        min_values = self.data.min()
        max_values = self.data.max()
        
        # Create the MissingValuesDialog instance
        dialog = MissingValuesDialog(missing_values, min_values, max_values, total_rows, parent=self)
        
        # Show the dialog
        dialog.exec_()


    def handle_preparation_click(self):
        dataset = self.merged_data if self.merged_data is not None else self.data
        dialog = PrepareDataDialog(dataset, self.file_name, self.main_window, self)
        if dialog.exec_():
            # Get the prepared data from the dialog
            prepared_data = dialog.get_prepared_data()

            # Update the data in this DataWindow
            self.data = prepared_data

            # If this is a merged dataset, update the merged_data as well
            if self.merged_data is not None:
                self.merged_data = prepared_data

            # Update the table to reflect the changes
            self.update_table()

            QMessageBox.information(self, "Data Preparation", "Data preparation completed successfully.")

    def drop_column_row(self):
        selected_items = self.table.selectedItems()
        selected_cols = list(set(item.column() for item in selected_items))
        selected_rows = list(set(item.row() for item in selected_items))
        self.data = self.data.drop(index=selected_rows, columns=self.data.columns[selected_cols])
        self.update_table()

    def normalize_data(self):
        # Filter numeric columns
        numeric_columns = self.data.select_dtypes(include=['number']).columns

        if not numeric_columns.empty:
            # Normalize only numeric columns
            self.data[numeric_columns] = (self.data[numeric_columns] - self.data[numeric_columns].min()) / \
                                        (self.data[numeric_columns].max() - self.data[numeric_columns].min())
            self.update_table()
        else:
            QMessageBox.warning(self, "Normalization Error", "No numeric columns found for normalization.")