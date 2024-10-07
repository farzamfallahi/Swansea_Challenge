
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

    def handle_save_click(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if file_path:
                try:
                    # Save the current data, whether it's merged or not
                    data_to_save = self.merged_data if self.merged_data is not None else self.data
                    data_to_save.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Save Success", f"Dataset saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")

    def handle_decision_tree_click(self):
        print("Decision tree button clicked manually")
        print("Decision tree visualization triggered in handle_decision_tree_click method")
        if self.data.shape[1] >= 2:
            params_dialog = DecisionTreeParamsDialog(self.data.columns.tolist(), self)
            if params_dialog.exec_():
                params = params_dialog.get_params()

                # Check if this dialog info already exists
                dialog_info = DialogInfo("DecisionTree", params)
                existing_dialog = next((d for d in self.dialog_history if d == dialog_info), None)
                if existing_dialog:
                    existing_dialog.instance_count += 1
                    dialog_info = existing_dialog  # Use the existing dialog info
                else:
                    self.dialog_history.append(dialog_info)

                print(f"Creating Decision Tree (instance {dialog_info.instance_count})")
                print("Params returned from dialog:", params)
                
                train_columns = params.get('train_columns', [])
                test_column = params.get('test_column')
                
                print("Train columns:", train_columns)
                print("Test column:", test_column)
                
                if not train_columns:
                    QMessageBox.warning(self, "Invalid Selection", "Please select at least one column for training.")
                    return
                
                if not test_column:
                    QMessageBox.warning(self, "Invalid Selection", "Please select a column for testing (target variable).")
                    return
                
                if test_column in train_columns:
                    QMessageBox.warning(self, "Invalid Selection", "Test column cannot be in training columns.")
                    return
                
                X = self.data[train_columns]
                y = self.data[test_column]
                
                # Data preparation (as before)
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object', 'category']).columns

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', numeric_features),
                        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
                    ])

                X_processed = preprocessor.fit_transform(X)

                onehot_encoder = preprocessor.named_transformers_['cat']
                cat_feature_names = [f"{feature}_{cat}" for feature, cats in zip(categorical_features, onehot_encoder.categories) for cat in cats[1:]]
                feature_names = list(numeric_features) + cat_feature_names

                if pd.api.types.is_numeric_dtype(y):
                    unique_values = y.nunique()
                    is_continuous = unique_values > 10
                else:
                    is_continuous = False
                    unique_values = y.nunique()
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)

                if is_continuous:
                    tree_type = "regression"
                    criterion = "squared_error"
                else:
                    tree_type = "classification"
                    criterion = "gini"

                target_labels = [str(label) for label in np.unique(y)]

                if params["use_grid_search"]:
                    param_grid = params.get("grid_search_params", {})
                    
                    if not param_grid:
                        QMessageBox.warning(self, "Invalid Input", "No valid grid search parameters provided.")
                        return

                    # Create the base model
                    if tree_type == "classification":
                        base_model = DecisionTreeClassifier(random_state=42)
                    else:
                        base_model = DecisionTreeRegressor(random_state=42)

                    # Fit the best model
                    grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
                    grid_search.fit(X_processed, y)
                    
                    best_params = grid_search.best_params_
                    best_tree = grid_search.best_estimator_
                    
                    # Create and show the best tree visualization
                    best_tree_widget = self.create_tree_visualization(best_tree, X_processed, y, feature_names, target_labels, "Best Decision Tree")
                    
                    # Create a dialog to display the best tree and add a button to show all trees
                    tree_dialog = QDialog(self)
                    tree_dialog.setWindowTitle("Decision Tree Visualization")
                    tree_layout = QVBoxLayout(tree_dialog)
                    
                    # Display best parameters
                    best_params_text = QTextEdit()
                    best_params_text.setPlainText(f"Best parameters: {best_params}")
                    tree_layout.addWidget(best_params_text)
                    
                    tree_layout.addWidget(best_tree_widget)
                    
                    # Add button to show all trees
                    show_all_button = QPushButton("Show All Trees")
                    show_all_button.clicked.connect(lambda: self.show_all_trees(grid_search, X_processed, y, feature_names, target_labels))
                    tree_layout.addWidget(show_all_button)
                    
                    tree_dialog.setLayout(tree_layout)
                    tree_dialog.resize(1200, 800)
                    tree_dialog.show()
                else:
                    # Single tree visualization (existing code)
                    tree_params = {
                        "max_depth": params["max_depth"],
                        "min_samples_split": params["min_samples_split"],
                        "min_samples_leaf": params["min_samples_leaf"],
                        "criterion": criterion
                    }
                    
                    if tree_type == "classification":
                        clf = DecisionTreeClassifier(random_state=42, **tree_params)
                    else:
                        clf = DecisionTreeRegressor(random_state=42, **tree_params)

                    clf.fit(X_processed, y)

                    # Create and show the decision tree dialog
                    decision_tree_dialog = QDialog(self)
                    decision_tree_dialog.setWindowTitle("Decision Tree Visualization")
                    decision_tree_layout = QVBoxLayout(decision_tree_dialog)

                    # Create matplotlib figure
                    fig, ax = plt.subplots(figsize=(15, 10))
                    plot_tree(clf, feature_names=feature_names, class_names=target_labels,
                            rounded=True, filled=True, fontsize=8, ax=ax,
                            proportion=True, precision=2, impurity=False, node_ids=True)
                    plt.title(f"{'Decision Tree Classifier' if tree_type == 'classification' else 'Decision Tree Regressor'} for {test_column}", fontsize=16, fontweight='bold')
                    plt.tight_layout(pad=1.0)

                    # Add matplotlib figure to dialog
                    canvas = FigureCanvas(fig)
                    decision_tree_layout.addWidget(canvas)

                    toolbar = NavigationToolbar(canvas, decision_tree_dialog)
                    decision_tree_layout.addWidget(toolbar)

                    decision_tree_dialog.setLayout(decision_tree_layout)
                    decision_tree_dialog.resize(1200, 800)
                    decision_tree_dialog.show()
        else:
            QMessageBox.warning(self, "Insufficient Columns", "The dataset should have at least two columns to create a decision tree.")

    def show_all_trees(self, grid_search, X, y, feature_names, target_labels):
        all_trees_dialog = QDialog(self)
        all_trees_layout = QVBoxLayout(all_trees_dialog)

        tab_widget = QTabWidget()

        for params in grid_search.cv_results_['params']:
            # Create and fit a new estimator with these parameters
            if isinstance(grid_search.estimator, DecisionTreeClassifier):
                estimator = DecisionTreeClassifier(**params, random_state=42)
            else:
                estimator = DecisionTreeRegressor(**params, random_state=42)
            estimator.fit(X, y)

            tree_widget = self.create_tree_visualization(estimator, X, y, feature_names, target_labels, f"Decision Tree")
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            tab_widget.addTab(tree_widget, param_str)

        all_trees_layout.addWidget(tab_widget)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(all_trees_dialog.close)
        all_trees_layout.addWidget(close_button)

        all_trees_dialog.setLayout(all_trees_layout)
        all_trees_dialog.resize(1000, 800)
        all_trees_dialog.exec_()

    def create_tree_visualization(self, tree, X, y, feature_names, target_labels, title):
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)

        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(tree, feature_names=feature_names, class_names=target_labels,
                rounded=True, filled=True, fontsize=8, ax=ax,
                proportion=True, precision=2, impurity=False, node_ids=True)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout(pad=1.0)

        canvas = FigureCanvas(fig)
        tree_layout.addWidget(canvas)

        toolbar = NavigationToolbar(canvas, tree_widget)
        tree_layout.addWidget(toolbar)

        return tree_widget   
        
    def handle_neural_network_click(self):
        columns = self.data.columns.tolist()
        params_dialog = NeuralNetworkParamsDialog(self, columns=columns)
        if params_dialog.exec_():
            params = params_dialog.get_params()
            if params is None:  # Check if params is None due to invalid input
                return
            try:
                result = prepare_and_train_neural_network(self.data, params)

                # Create and show NeuralNetworkResults
                results_window = NeuralNetworkResults(self.data, params['test_columns'], params, result, parent=self)
                results_window.exec_()  # Use exec_() instead of show() for modal behavior

            except ValueError as e:
                QMessageBox.critical(self, "Error", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

    def show_lime_results(self, model, X_test, feature_names):
        lime_window = QDialog(self)
        lime_layout = QVBoxLayout()

        # Convert X_test to numpy array if it's a DataFrame
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test

        # For binary classification, we need to modify the predict function
        def predict_fn(x):
            # Get the probabilities
            proba = model.predict_proba(x)
            
            # Convert list to numpy array if necessary
            if isinstance(proba, list):
                proba = np.array(proba)
                
            # If proba has 3 dimensions, squeeze it to 2D
            if proba.ndim == 3:
                proba = np.squeeze(proba, axis=0)
                
            # Ensure the output is 2D with two columns (binary classification)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
                
            if proba.shape[1] == 1:
                proba = np.hstack([1 - proba, proba])
                
            return proba

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test_array,
            feature_names=feature_names,
            class_names=['Class 0', 'Class 1'],  # Binary classification
            mode='classification'
        )

        instance_idx = np.random.randint(0, X_test_array.shape[0])
        instance = X_test_array[instance_idx]

        exp = explainer.explain_instance(
            instance, 
            predict_fn,
            num_features=10,
            num_samples=5000
        )

        text_explanation = QTextEdit()
        text_explanation.setReadOnly(True)
        text_explanation.setHtml("<h3>LIME Explanation:</h3>")
        
        # Get available labels from the explanation
        available_labels = exp.available_labels()
        
        for label in available_labels:
            text_explanation.append(f"<h4>Explanation for Class {label}:</h4>")
            try:
                for feature, importance in exp.as_list(label=label):
                    text_explanation.append(f"<b>{feature}:</b> {importance:.4f}<br>")
            except KeyError:
                text_explanation.append("No explanation available for this class.<br>")
            text_explanation.append("<br>")
        
        lime_layout.addWidget(text_explanation)

        # Add LIME plot
        try:
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            canvas = FigureCanvas(plt.gcf())
            lime_layout.addWidget(canvas)
        except Exception as e:
            error_text = QLabel(f"Error generating LIME plot: {str(e)}")
            lime_layout.addWidget(error_text)

        lime_window.setLayout(lime_layout)
        lime_window.setWindowTitle("LIME Results")
        lime_window.resize(800, 600)
        lime_window.show()

    def show_shap_results(self, model, X_test, feature_names):
        shap_window = QDialog(self)
        shap_layout = QVBoxLayout()

        try:
            # Get SHAP values
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)

            # Create SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            
            canvas = FigureCanvas(plt.gcf())
            shap_layout.addWidget(canvas)

            # Add textual explanation
            text_explanation = QTextEdit()
            text_explanation.setReadOnly(True)
            text_explanation.setHtml("<h3>SHAP Explanation:</h3>")

            mean_shap_values = np.abs(shap_values.values).mean(axis=0)
            feature_importance = sorted(zip(feature_names, mean_shap_values), key=lambda x: abs(x[1]), reverse=True)

            for feature, importance in feature_importance:
                text_explanation.append(f"<b>{feature}:</b> {float(importance):.4f}<br>")

            shap_layout.addWidget(text_explanation)

        except Exception as e:
            error_message = QLabel(f"Error in SHAP computation: {str(e)}")
            shap_layout.addWidget(error_message)

        shap_window.setLayout(shap_layout)
        shap_window.setWindowTitle("SHAP Results")
        shap_window.resize(800, 600)
        shap_window.show()

    def show_eli5_results(self, model, X, y, feature_names):
        eli5_window = QDialog(self)
        eli5_layout = QVBoxLayout()

        try:
            # Get ELI5 explanation
            explanation = get_eli5_explanation(model, X, y, mode='weights', feature_names=feature_names)
        except Exception as e:
            explanation = f"An error occurred while generating the ELI5 explanation: {str(e)}\n\n"
            explanation += f"Model type: {type(model)}\n"
            explanation += f"X shape: {X.shape}\n"
            explanation += f"y shape: {y.shape}\n"
        
        # Display ELI5 explanation
        eli5_text = QTextEdit()
        eli5_text.setReadOnly(True)
        eli5_text.setPlainText(explanation)
        eli5_layout.addWidget(eli5_text)
        eli5_window.setLayout(eli5_layout)
        eli5_window.setWindowTitle("ELI5 Explanation")
        eli5_window.resize(800, 600)
        eli5_window.show()

    def show_confusion_matrix(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        confusion_matrix_window = QDialog(self)
        confusion_matrix_window.setWindowTitle("Confusion Matrix")
        layout = QVBoxLayout()

        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        confusion_matrix_window.setLayout(layout)
        confusion_matrix_window.resize(800, 600)
        confusion_matrix_window.show()

    def handle_knn_click(self):
        print("Starting KNN training")
        columns = self.data.columns.tolist()
        params_dialog = KNNParamsDialog(self, columns=columns)
        if params_dialog.exec_():
            params = params_dialog.get_params()


            # Check if this dialog info already exists
            dialog_info = DialogInfo("KNN", params)
            if dialog_info in self.dialog_history:
                QMessageBox.information(self, "Duplicate Dialog", "This KNN configuration has already been used.")
                return
            
            # Remove any existing decision tree dialogs
            self.dialog_history = [di for di in self.dialog_history if di.dialog_type != "DecisionTree"]
            
            self.dialog_history.append(dialog_info)

            
            if not params['train_columns']:
                QMessageBox.warning(self, "Invalid Selection", "Please select at least one column for training.")
                return
            
            if not params['test_column']:
                QMessageBox.warning(self, "Invalid Selection", "Please select a target column.")
                return
            
            if params['test_column'] in params['train_columns']:
                QMessageBox.warning(self, "Invalid Selection", "Target column cannot be in training columns.")
                return
            
            # Prepare and train the KNN model
            model, X_train, X_test, y_train, y_test, preprocessor = prepare_and_train_knn(self.data, params)

            # Create a new window to display results
            results_window = QDialog(self)
            results_window.setWindowTitle("K-Nearest Neighbors Results")
            results_layout = QVBoxLayout()

            # Display metrics
            results_text = QTextEdit()
            results_text.setReadOnly(True)
            
            if isinstance(model, KNeighborsClassifier):
                accuracy = accuracy_score(y_test, model.predict(X_test))
                f1 = f1_score(y_test, model.predict(X_test), average='weighted')
                results_text.setHtml(f"""
                    <b>Accuracy:</b> {accuracy:.4f}<br>
                    <b>F1 Score:</b> {f1:.4f}<br>
                """)
            elif isinstance(model, KNeighborsRegressor):
                mse = mean_squared_error(y_test, model.predict(X_test))
                r2 = r2_score(y_test, model.predict(X_test))
                results_text.setHtml(f"""
                    <b>Mean Squared Error:</b> {mse:.4f}<br>
                    <b>R-squared Score:</b> {r2:.4f}<br>
                """)
            
            results_layout.addWidget(results_text)

            # Add LIME button
            lime_button = QPushButton("Show LIME Results")
            lime_button.clicked.connect(lambda: self.show_knn_lime_results(model, X_train, X_test, preprocessor))
            results_layout.addWidget(lime_button)

            # Add SHAP button
            shap_button = QPushButton("Show SHAP Results")
            shap_button.clicked.connect(lambda: self.show_knn_shap_results(model, X_test, preprocessor))
            results_layout.addWidget(shap_button)

            # Add ELI5 button
            eli5_button = QPushButton("Show ELI5 Explanation")
            eli5_button.clicked.connect(lambda: self.show_eli5_results(model, X_test, y_test, preprocessor.get_feature_names_out()))
            results_layout.addWidget(eli5_button)

            # Add Time Series LIME button
            time_series_lime_button = QPushButton("Show Time Series LIME Results")
            time_series_lime_button.clicked.connect(lambda: show_time_series_lime_results_knn(model, X_train, 0, preprocessor.get_feature_names_out()))
            results_layout.addWidget(time_series_lime_button)

            results_window.setLayout(results_layout)
            results_window.resize(800, 600)
            results_window.show()
        print("Finished KNN training")

    def handle_svm_click(self):
        columns = self.data.columns.tolist()
        params_dialog = SVMParamsDialog(self, columns=columns)
        if params_dialog.exec_():
            params = params_dialog.get_params()

            if not params['train_columns']:
                QMessageBox.warning(self, "Invalid Selection", "Please select at least one column for training.")
                return

            if not params['test_column']:
                QMessageBox.warning(self, "Invalid Selection", "Please select a target column.")
                return

            if params['test_column'] in params['train_columns']:
                QMessageBox.warning(self, "Invalid Selection", "Target column cannot be in training columns.")
                return

            # Prepare and train the SVM model
            X = self.data[params['train_columns']]
            y = self.data[params['test_column']]
            model, X_train, X_test, y_train, y_test, preprocessor = prepare_and_train_svm(X, y, params)

            # Create a new window to display results
            results_window = QDialog(self)
            results_window.setWindowTitle("Support Vector Machine Results")
            results_layout = QVBoxLayout()

            # Display metrics
            results_text = QTextEdit()
            results_text.setReadOnly(True)

            accuracy = accuracy_score(y_test, model.predict(X_test))
            f1 = f1_score(y_test, model.predict(X_test), average='weighted')
            results_text.setHtml(f"""
                <b>Accuracy:</b> {accuracy:.4f}<br>
                <b>F1 Score:</b> {f1:.4f}<br>
            """)
            results_layout.addWidget(results_text)

            # Add LIME button
            lime_button = QPushButton("Show LIME Results")
            lime_button.clicked.connect(lambda: self.show_svm_lime_results(model, X_train, X_test, y_test, preprocessor))
            results_layout.addWidget(lime_button)

            # Add SHAP button
            shap_button = QPushButton("Show SHAP Results")
            shap_button.clicked.connect(lambda: self.show_svm_shap_results(model, X_test, preprocessor))
            results_layout.addWidget(shap_button)

            # Add ELI5 button
            eli5_button = QPushButton("Show ELI5 Explanation")
            eli5_button.clicked.connect(lambda: self.show_eli5_results(model, X_test, y_test, preprocessor.get_feature_names_out()))
            results_layout.addWidget(eli5_button)

            # Replace the existing Time Series LIME button with this new one
            time_series_lime_button = QPushButton("Show Time Series LIME Results")
            time_series_lime_button.clicked.connect(lambda: show_time_series_lime_results_svm(model, X_train, 0, X.columns.tolist()))
            results_layout.addWidget(time_series_lime_button)

            results_window.setLayout(results_layout)
            results_window.resize(800, 600)
            results_window.show()

    def handle_gbm_click(self):
        columns = self.data.columns.tolist()
        params_dialog = GradientBoostingParamsDialog(self, columns=columns)
        if params_dialog.exec_():
            params = params_dialog.get_params()

            try:
                # Change this line to capture X_train
                model, X_train, X_test, y_train, y_test, preprocessor = prepare_and_train_gbm(self.data, params)

                # Create a new window to display results
                results_window = QDialog(self)
                results_window.setWindowTitle("Gradient Boosting Machine Results")
                results_layout = QVBoxLayout()

                # Display metrics
                results_text = QTextEdit()
                results_text.setReadOnly(True)

                if isinstance(model, GradientBoostingClassifier):
                    accuracy = accuracy_score(y_test, model.predict(X_test))
                    f1 = f1_score(y_test, model.predict(X_test), average='weighted')
                    results_text.setHtml(f"""
                        <b>Accuracy:</b> {accuracy:.4f}<br>
                        <b>F1 Score:</b> {f1:.4f}<br>
                    """)
                elif isinstance(model, GradientBoostingRegressor):
                    mse = mean_squared_error(y_test, model.predict(X_test))
                    r2 = r2_score(y_test, model.predict(X_test))
                    results_text.setHtml(f"""
                        <b>Mean Squared Error:</b> {mse:.4f}<br>
                        <b>R-squared Score:</b> {r2:.4f}<br>
                    """)

                results_layout.addWidget(results_text)

                # Add LIME button
                lime_button = QPushButton("Show LIME Results")
                lime_button.clicked.connect(lambda: self.show_gbm_lime_results(model, X_test, preprocessor))
                results_layout.addWidget(lime_button)

                # Add SHAP button
                shap_button = QPushButton("Show SHAP Results")
                shap_button.clicked.connect(lambda: self.show_gbm_shap_results(model, X_test, preprocessor))
                results_layout.addWidget(shap_button)

                # In the handle_gbm_click method of your DataWindow class
                time_series_lime_button = QPushButton("Show Time Series LIME Results")
                time_series_lime_button.clicked.connect(lambda: show_time_series_lime_results_gbm(model, X_train, 0, preprocessor.get_feature_names_out()))
                results_layout.addWidget(time_series_lime_button)

                results_window.setLayout(results_layout)
                results_window.resize(800, 600)
                results_window.show()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def handle_logistic_regression_click(self):
        columns = self.data.columns.tolist()
        params_dialog = LogisticRegressionParamsDialog(self, columns=columns)
        if params_dialog.exec_():
            params = params_dialog.get_params()
            print("Params received:", params)  # Debug print

            if not params['train_columns']:
                QMessageBox.warning(self, "Invalid Selection", "Please select at least one training column.")
                return

            if not params['test_columns']:
                QMessageBox.warning(self, "Invalid Selection", "Please select at least one test column.")
                return

            try:
                print("Train columns:", params['train_columns'])
                print("Test columns:", params['test_columns'])
                print("Available columns:", self.data.columns)

                X = self.data[params['train_columns']]
                y = self.data[params['test_columns']]

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=42)

                # Create a preprocessing pipeline
                preprocessor = Pipeline([
                    ('scaler', StandardScaler()),
                ])

                # Create the model
                base_model = LogisticRegression(C=params['C'], max_iter=params['max_iter'])
                model = MultiOutputClassifier(base_model)

                # Create the full pipeline
                full_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', MultiOutputClassifier(LogisticRegression(C=params['C'], max_iter=params['max_iter'])))
                ])

                # Fit the model
                full_pipeline.fit(X_train, y_train)

                # Predictions
                y_pred = full_pipeline.predict(X_test)

                # Create a new window to display results
                results_window = QDialog(self)
                results_window.setWindowTitle("Logistic Regression Results")
                results_layout = QVBoxLayout()

                # Display metrics
                results_text = QTextEdit()
                results_text.setReadOnly(True)

                # Calculate metrics for each output column
                accuracies = []
                f1_scores = []
                for i in range(y_test.shape[1]):
                    accuracies.append(accuracy_score(y_test.iloc[:, i], y_pred[:, i]))
                    f1_scores.append(f1_score(y_test.iloc[:, i], y_pred[:, i], average='weighted'))

                results_html = "<h3>Results:</h3>"
                for i, col in enumerate(params['test_columns']):
                    results_html += f"<b>{col}:</b><br>"
                    results_html += f"Accuracy: {accuracies[i]:.4f}<br>"
                    results_html += f"F1 Score: {f1_scores[i]:.4f}<br><br>"

                results_text.setHtml(results_html)
                results_layout.addWidget(results_text)

                # Add LIME button
                lime_button = QPushButton("Show LIME Results")
                lime_button.clicked.connect(lambda: self.show_lime_results(full_pipeline, X_test, params['train_columns']))
                results_layout.addWidget(lime_button)

                # Add SHAP button
                shap_button = QPushButton("Show SHAP Results")
                shap_button.clicked.connect(lambda: self.show_shap_results(full_pipeline, X_test, params['train_columns']))
                results_layout.addWidget(shap_button)

                results_window.setLayout(results_layout)
                results_window.resize(800, 600)
                results_window.show()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
                return

    def show_shap_results(self, model, X_test, feature_names):
        shap_window = QDialog(self)
        shap_layout = QVBoxLayout()

        debug_text = QTextEdit()
        debug_text.setReadOnly(True)
        shap_layout.addWidget(debug_text)

        def log(message):
            debug_text.append(message)
            print(message)  # Also print to console for debugging

        try:
            log("Starting feature importance analysis...")

            log(f"Model type: {type(model)}")
            log(f"X_test type: {type(X_test)}")
            log(f"X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}")
            log(f"Number of feature names: {len(feature_names)}")

            if isinstance(X_test, pd.DataFrame):
                X_test_array = X_test.values
                log("Converted X_test from DataFrame to numpy array")
            else:
                X_test_array = X_test
            
            log(f"X_test_array shape: {X_test_array.shape}")

            log("Attempting to get baseline predictions...")
            baseline_predictions = model.predict(X_test_array)
            log(f"Baseline predictions shape: {baseline_predictions.shape}")

            log("Calculating feature importance...")
            feature_importance = []
            for i in range(X_test_array.shape[1]):
                log(f"Processing feature {i + 1} of {X_test_array.shape[1]}")
                X_permuted = X_test_array.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_predictions = model.predict(X_permuted)
                
                if baseline_predictions.ndim == 2 and baseline_predictions.shape[1] > 1:
                    importance = np.mean([np.mean((baseline_predictions[:, j] - permuted_predictions[:, j])**2) 
                                        for j in range(baseline_predictions.shape[1])])
                else:
                    importance = np.mean((baseline_predictions - permuted_predictions)**2)
                
                feature_importance.append(importance)
                log(f"Importance for feature {i + 1}: {importance}")

            log("Normalizing feature importance...")
            feature_importance = np.array(feature_importance) / np.sum(feature_importance)

            log("Sorting features by importance...")
            sorted_idx = np.argsort(feature_importance)
            sorted_features = [feature_names[i] for i in sorted_idx]

            log("Generating bar plot...")
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_features)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Normalized Feature Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            canvas = FigureCanvas(plt.gcf())
            shap_layout.addWidget(canvas)

            log("Generating text explanation...")
            text_explanation = QTextEdit()
            text_explanation.setReadOnly(True)
            text_explanation.setHtml("<h3>Feature Importance Explanation:</h3>")

            for feature, importance in zip(sorted_features[::-1], feature_importance[sorted_idx][::-1]):
                text_explanation.append(f"<b>{feature}:</b> {importance:.4f}<br>")

            shap_layout.addWidget(text_explanation)

            log("Feature importance analysis completed successfully.")

        except Exception as e:
            log(f"Error in feature importance computation: {str(e)}")
            error_message = QLabel(f"Error in feature importance computation: {str(e)}")
            shap_layout.addWidget(error_message)

        shap_window.setLayout(shap_layout)
        shap_window.setWindowTitle("Feature Importance Results")
        shap_window.resize(800, 600)
        shap_window.show()

    def show_gbm_lime_results(self, model, X_test, preprocessor):
        lime_window = QDialog(self)
        lime_layout = QVBoxLayout()

        # Add a QTextEdit for logging
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        lime_layout.addWidget(log_text)

        def log(message):
            log_text.append(message)
            print(message)  # Also print to console for debugging

        try:
            log("Starting LIME analysis...")

            # Get feature names from the preprocessor
            feature_names = preprocessor.get_feature_names_out()
            log(f"Number of features in X_test: {X_test.shape[1]}")
            log(f"Number of feature names: {len(feature_names)}")
            log(f"Feature names: {feature_names}")

            # Get class names from the target variable
            class_names = np.unique(self.data[self.data.columns[-1]])
            log(f"Class names: {class_names}")

            # Ensure X_test is a numpy array
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
            log(f"X_test shape: {X_test.shape}")

            # Handle feature mismatch
            if X_test.shape[1] != len(feature_names):
                log("Warning: Mismatch between number of features and feature names.")
                log("Attempting to use the preprocessor to transform X_test...")
                X_test = preprocessor.transform(X_test)
                log(f"X_test shape after transformation: {X_test.shape}")

            if X_test.shape[1] != len(feature_names):
                log("Error: Feature mismatch persists. Using generic feature names.")
                feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

            log("Creating LIME explainer...")
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test,
                feature_names=feature_names,
                class_names=class_names,
                discretize_continuous=True
            )

            # Select a random instance to explain
            instance_index = np.random.randint(X_test.shape[0])
            log(f"Explaining instance at index: {instance_index}")
            exp = explainer.explain_instance(
                X_test[instance_index], 
                model.predict_proba, 
                num_features=10
            )

            log("Creating LIME plot...")
            # Manually create the LIME plot with orange bars
            fig, ax = plt.subplots(figsize=(12, 8))
            features, values = zip(*exp.as_list())
            y_pos = range(len(features))
            
            # Create orange bars
            bars = ax.barh(y_pos, values, align='center', color='orange', alpha=0.8)
            
            # Add value labels to the end of each bar
            for i, v in enumerate(values):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            ax.set_title('LIME Explanation', fontsize=14, fontweight='bold')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a light grid
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Adjust layout and display
            plt.tight_layout()
            
            log("Adding plot to window...")
            canvas = FigureCanvas(fig)
            lime_layout.addWidget(canvas)

            # Add textual explanation
            log("Adding textual explanation...")
            text_explanation = QTextEdit()
            text_explanation.setReadOnly(True)
            text_explanation.setHtml("<h3>LIME Explanation:</h3>")
            for feature, importance in exp.as_list():
                text_explanation.append(f"<b>{feature}:</b> {importance:.4f}<br>")
            lime_layout.addWidget(text_explanation)

            log("LIME analysis completed successfully.")

        except Exception as e:
            log(f"Error in LIME computation: {str(e)}")
            error_message = QLabel(f"Error in LIME computation: {str(e)}")
            lime_layout.addWidget(error_message)

        lime_window.setLayout(lime_layout)
        lime_window.setWindowTitle("GBM LIME Results")
        lime_window.resize(1000, 800)  # Increased size for better visibility
        lime_window.show()

    def show_gbm_shap_results(self, model, X_test, preprocessor):
        shap_window = QDialog(self)
        shap_layout = QVBoxLayout()

        try:
            # Get feature names from the preprocessor
            feature_names = preprocessor.get_feature_names_out()

            # Ensure X_test is a DataFrame with correct columns
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=feature_names)
            else:
                # If X_test is already a DataFrame, ensure it has all required columns
                missing_columns = set(feature_names) - set(X_test.columns)
                for col in missing_columns:
                    X_test[col] = 0  # Fill missing columns with a default value

            # Ensure X_test has columns in the correct order
            X_test = X_test[feature_names]

            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test)

            # Create SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            
            canvas = FigureCanvas(fig)
            shap_layout.addWidget(canvas)

            # Add text explanation
            text_explanation = QTextEdit()
            text_explanation.setReadOnly(True)
            text_explanation.setHtml("<h3>SHAP Explanation:</h3>")

            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(shap_values.values).mean(axis=0)

            # Sort features by importance
            feature_importance = sorted(zip(feature_names, mean_shap_values), key=lambda x: x[1], reverse=True)

            for feature, importance in feature_importance:
                text_explanation.append(f"<b>{feature}:</b> {importance:.4f}<br>")

            shap_layout.addWidget(text_explanation)

        except Exception as e:
            error_message = QLabel(f"Error in SHAP computation: {str(e)}")
            shap_layout.addWidget(error_message)

        shap_window.setLayout(shap_layout)
        shap_window.setWindowTitle("GBM SHAP Results")
        shap_window.resize(800, 600)
        shap_window.show()
    
    def show_svm_shap_results(self, model, X_test, preprocessor):
        feature_names = preprocessor.get_feature_names_out()
        
        print(f"X_test shape: {X_test.shape}")
        print(f"Number of feature names: {len(feature_names)}")

        # Ensure X_test is 2D
        if X_test.ndim != 2:
            X_test = X_test.reshape(-1, len(feature_names))

        if X_test.shape[1] != len(feature_names):
            print(f"Mismatch: X_test has {X_test.shape[1]} features, but {len(feature_names)} feature names provided.")
            return

        shap_window = QDialog(self)
        shap_layout = QVBoxLayout()

        try:
            # If X_test is sparse, convert to dense
            if scipy.sparse.issparse(X_test):
                X_test = X_test.toarray()

            shap_values, fig = perform_svm_shap_analysis(model, X_test, feature_names)
            canvas = FigureCanvas(fig)
            shap_layout.addWidget(canvas)
        except Exception as e:
            error_message = QLabel(f"Error in SHAP computation: {str(e)}")
            shap_layout.addWidget(error_message)
            print(f"Detailed error: {e}")
            print(f"X_test type: {type(X_test)}")
            print(f"X_test shape after potential reshaping: {X_test.shape}")

        shap_window.setLayout(shap_layout)
        shap_window.setWindowTitle("SVM SHAP Results")
        shap_window.resize(800, 600)
        shap_window.show()

    def show_svm_lime_results(self, model, X_train, X_test, y_test, preprocessor):
        lime_window = QDialog(self)
        lime_layout = QVBoxLayout()

        try:
            # Get feature names from the preprocessor
            feature_names = preprocessor.get_feature_names_out()

            # Convert to dense array if sparse
            if scipy.sparse.issparse(X_train):
                X_train = X_train.toarray()
            if scipy.sparse.issparse(X_test):
                X_test = X_test.toarray()

            # Ensure X_train and X_test are numpy arrays
            X_train = np.array(X_train)
            X_test = np.array(X_test)

            # Determine if it's a classification or regression task
            if isinstance(model, SVC):
                mode = 'classification'
                predict_fn = model.predict_proba
                class_names = list(model.classes_)
            else:
                mode = 'regression'
                predict_fn = model.predict
                class_names = None

            # Perform LIME analysis
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=class_names,
                mode=mode,
                random_state=42
            )

            # Select a single instance for explanation
            instance_index = 0
            instance = X_test[instance_index]

            # Generate the explanation
            exp = explainer.explain_instance(
                instance, 
                predict_fn,
                num_features=len(feature_names),
                num_samples=5000
            )

            # Get explanation as a list and sort by absolute importance
            explanation = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)

            # Display LIME explanation text
            lime_text = QTextEdit()
            lime_text.setReadOnly(True)
            lime_text.setHtml("<h3>LIME Explanation:</h3>")
            for feature, importance in explanation:
                lime_text.append(f"<b>{feature}:</b> {importance:.6f}<br>")
            lime_layout.addWidget(lime_text)

            # Generate and display LIME plot
            fig, ax = plt.subplots(figsize=(10, 8))
            features, importances = zip(*explanation)
            y_pos = range(len(features))
            
            # Create color list based on importance values
            colors = ['orange' if imp >= 0 else 'blue' for imp in importances]
            
            ax.barh(y_pos, importances, align='center', color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Feature Importance')
            ax.set_title('LIME Explanation for SVM')
            
            # Adjust x-axis to show small values
            max_importance = max(abs(imp) for imp in importances)
            ax.set_xlim(-max_importance, max_importance)
            
            # Add a vertical line at x=0 for better readability
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()

            canvas = FigureCanvas(fig)
            lime_layout.addWidget(canvas)

            # Add toolbar for the plot
            toolbar = NavigationToolbar(canvas, lime_window)
            lime_layout.addWidget(toolbar)

        except Exception as e:
            error_message = f"An error occurred while generating the LIME plot: {str(e)}"
            print(error_message)
            lime_text = QTextEdit()
            lime_text.setPlainText(error_message)
            lime_layout.addWidget(lime_text)

        lime_window.setLayout(lime_layout)
        lime_window.setWindowTitle("SVM LIME Results")
        lime_window.resize(800, 600)
        lime_window.show()

    def show_knn_shap_results(self, model, X_test, preprocessor):
        shap_window = QDialog(self)
        shap_layout = QVBoxLayout()

        # Get feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()

        # Compute SHAP values
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test)

        # Create SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show(block=False)

        canvas = FigureCanvas(fig)
        shap_layout.addWidget(canvas)

        shap_window.setLayout(shap_layout)
        shap_window.setWindowTitle("KNN SHAP Results")
        shap_window.resize(800, 600)
        shap_window.show()

    def show_knn_lime_results(self, model, X_train, X_test, preprocessor):
        lime_window = QDialog(self)
        lime_layout = QVBoxLayout()

        # Get feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()

        # Perform LIME analysis
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=model.classes_, discretize_continuous=True)
        exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=len(feature_names))

        # Display LIME explanation
        lime_text = QTextEdit()
        lime_text.setReadOnly(True)
        lime_text.setHtml("<h3>LIME Explanation:</h3>")
        for feature, importance in exp.as_list():
            lime_text.append(f"<b>{feature}:</b> {importance:.4f}<br>")
        lime_layout.addWidget(lime_text)

        # Display LIME plot
        fig = exp.as_pyplot_figure()
        canvas = FigureCanvas(fig)
        lime_layout.addWidget(canvas)

        lime_window.setLayout(lime_layout)
        lime_window.setWindowTitle("KNN LIME Results")
        lime_window.resize(800, 600)
        lime_window.show()

    def show_time_series_lime_results(self, model, X_train, instance_index, feature_names):
        lime_window = QDialog(self)
        lime_layout = QVBoxLayout()

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

        # Display the explanation
        lime_text = QTextEdit()
        lime_text.setReadOnly(True)
        lime_text.setPlainText(str(explanation.as_list()))
        lime_layout.addWidget(lime_text)

        lime_window.setLayout(lime_layout)
        lime_window.setWindowTitle("LIME Results")
        lime_window.show()

    def show_recommendation(self):
        recommendations = self.analyze_dataset()
        dialog = RecommendationDialog(recommendations, self)
        dialog.exec_()

    def analyze_dataset(self):
        recommendations = []
        num_samples, num_features = self.data.shape
        target_column = self.data.columns[-1]
        num_classes = self.data[target_column].nunique()
        has_missing_values = self.data.isnull().any().any()
        is_balanced = len(self.data[target_column].value_counts().unique()) == 1
        is_binary = num_classes == 2
        is_multiclass = num_classes > 2
        is_regression = self.data[target_column].dtype in ['float64', 'float32', 'int64', 'int32'] and num_classes > 10

        if is_regression:
            recommendations.append("This appears to be a regression problem.")
        elif is_binary:
            recommendations.append("This appears to be a binary classification problem.")
        elif is_multiclass:
            recommendations.append("This appears to be a multi-class classification problem.")

        if num_samples < 1000:
            recommendations.append("Your dataset is relatively small. Consider using models that work well with limited data:")
            recommendations.append("- K-Nearest Neighbors: Good for small to medium-sized datasets.")
            recommendations.append("- Decision Tree: Can work well on smaller datasets and provides interpretable results.")
            if is_binary or (is_multiclass and num_classes < 5):
                recommendations.append("- Support Vector Machines: Effective for smaller datasets, especially in binary classification.")

        if num_samples >= 1000:
            recommendations.append("Your dataset is of moderate to large size. Consider these models:")
            recommendations.append("- Random Forest: Handles large datasets well and is less prone to overfitting.")
            recommendations.append("- Gradient Boosting Machines: Powerful for both classification and regression tasks.")
            if num_samples > 10000 and num_features > 20:
                recommendations.append("- Neural Networks: Can capture complex patterns in large datasets with many features.")

        if is_binary:
            recommendations.append("For binary classification, also consider:")
            recommendations.append("- Logistic Regression: Simple and interpretable for binary problems.")

        if is_multiclass:
            recommendations.append("For multi-class problems, these models are particularly suitable:")
            recommendations.append("- Random Forest: Handles multi-class problems naturally.")
            recommendations.append("- Support Vector Machines (with 'one-vs-rest' strategy): Can be effective for multi-class classification.")

        if has_missing_values:
            recommendations.append("Your dataset contains missing values. Consider:")
            recommendations.append("- Decision Trees or Random Forests: Can handle missing values without imputation.")
            recommendations.append("- Gradient Boosting Machines: Can work with missing values in some implementations.")
            recommendations.append("Alternatively, consider imputing missing values before using other models.")

        if not is_balanced and (is_binary or is_multiclass):
            recommendations.append("Your dataset appears to be imbalanced. Consider:")
            recommendations.append("- Ensemble methods like Random Forest or Gradient Boosting: Often handle imbalanced data better.")
            recommendations.append("- Using techniques like oversampling, undersampling, or SMOTE to balance your dataset.")
            recommendations.append("- Adjusting class weights in models that support it (e.g., Logistic Regression, SVM).")

        if num_features > 100:
            recommendations.append("Your dataset has a high number of features. Consider:")
            recommendations.append("- Feature selection techniques to reduce dimensionality.")
            recommendations.append("- Principal Component Analysis (PCA) for dimensionality reduction.")
            recommendations.append("- Regularized models like Lasso or Ridge regression.")

        return recommendations
    
class RecommendationDialog(QDialog):
    def __init__(self, recommendations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Recommendations")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        # Create a QTextEdit widget to display the recommendations
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # Format the recommendations
        recommendation_text = "Based on the analysis of your dataset, here are the model recommendations:\n\n"
        for recommendation in recommendations:
            recommendation_text += f"• {recommendation}\n\n"
        
        self.text_edit.setPlainText(recommendation_text)

        layout.addWidget(self.text_edit)

        # Add an OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

        self.setLayout(layout)

class MultiDatasetMergeDialog(NonModalDialog):
    def __init__(self, data_windows, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.data_windows = data_windows  # Add this line
        self.selected_columns = {}
        self.checkboxes = {}
        self.setup_ui()

        self.merge_button = QPushButton("Merge Selected Columns")
        self.merge_button.clicked.connect(self.handle_merge)
        self.layout().addWidget(self.merge_button)  # Use self.layout() instead of layout

    def handle_merge(self):
        merged_data = self.get_merged_data()
        if merged_data is not None:
            new_window = DataWindow(merged_data, "Merged from multiple files", self.main_window, self)
            self.main_window.data_windows.append(new_window)
            new_window.show()
        else:
            QMessageBox.warning(self, "Merge Error", "No columns were selected for merging.")
        self.close()

    def setup_ui(self):
        self.setWindowTitle("Merge Columns from Multiple Datasets")
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for data_window in self.data_windows:
            group_box = QGroupBox(f"Dataset: {data_window.file_name}")
            group_layout = QVBoxLayout()
           
            select_all_button = QPushButton("Select All")
            select_all_button.clicked.connect(lambda checked, w=data_window: self.select_all_columns(w))
            group_layout.addWidget(select_all_button)
           
            self.checkboxes[data_window] = []
            for column in data_window.data.columns:
                checkbox = QCheckBox(str(column))
                checkbox.setProperty('column_name', column)
                checkbox.stateChanged.connect(lambda state, w=data_window, c=column: self.update_selection(w, c, state))
                group_layout.addWidget(checkbox)
                self.checkboxes[data_window].append(checkbox)
           
            group_box.setLayout(group_layout)
            scroll_layout.addWidget(group_box)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        merge_button = QPushButton("Merge Selected Columns")
        merge_button.clicked.connect(self.accept)
        layout.addWidget(merge_button)

        merge_rows_button = QPushButton("Merge from rows")
        merge_rows_button.clicked.connect(self.merge_from_rows)
        layout.addWidget(merge_rows_button)

        self.setLayout(layout)

    def select_all_columns(self, data_window):
        for checkbox in self.checkboxes[data_window]:
            checkbox.setChecked(True)

    def update_selection(self, data_window, column, state):
        if data_window not in self.selected_columns:
            self.selected_columns[data_window] = set()
       
        if state == Qt.Checked:
            self.selected_columns[data_window].add(column)
        else:
            self.selected_columns[data_window].discard(column)

    def get_merged_data(self):
        merged_data = pd.DataFrame()
        for data_window, columns in self.selected_columns.items():
            if columns:
                selected_data = data_window.data[list(columns)]
                merged_data = pd.concat([merged_data, selected_data], axis=1)
        return merged_data if not merged_data.empty else None

    def merge_from_rows(self):
        dfs_to_merge = []
        for data_window in self.data_windows:
            if data_window in self.selected_columns and self.selected_columns[data_window]:
                selected_columns = list(self.selected_columns[data_window])
                selected_data = data_window.data[selected_columns]
                print(f"Merging data from {data_window.file_name}:")
                print(selected_data)
                dfs_to_merge.append(selected_data)

        if dfs_to_merge:
            merged_data = pd.concat(dfs_to_merge, axis=0, ignore_index=True)
            new_window = DataWindow(merged_data, "Merged from rows", self.main_window)
            self.main_window.data_windows.append(new_window)
            new_window.show()  # Change this from new_window.exec_() to new_window.show()
        else:
            QMessageBox.warning(self, "Merge Error", "No datasets were selected for merging.")
        
        self.accept()

class ColumnTitleDialog(QDialog):
    def __init__(self, initial_title, num_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Column Title")
        self.setModal(True)
        self.num_columns = num_columns

        layout = QVBoxLayout()

        # Add input field for column title
        self.title_edit = QLineEdit()
        self.title_edit.setText(initial_title)
        layout.addWidget(self.title_edit)

        # Add input field for column order
        order_layout = QHBoxLayout()
        order_label = QLabel("Column Order (0 to {})".format(num_columns - 1))
        order_layout.addWidget(order_label)
        self.order_edit = QLineEdit()
        order_layout.addWidget(self.order_edit)
        layout.addLayout(order_layout)

        # Add button for modifying column title and order
        self.modify_btn = QPushButton("Modify")
        self.modify_btn.clicked.connect(self.accept)
        layout.addWidget(self.modify_btn)

        self.setLayout(layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XAI Application")
        self.resize(800, 600)
        self.setup_ui()
        self.datasets = []
        self.file_paths = []
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.merged_data = None
        self.dialog_history = []
        self.data_windows = []  # Make sure this line is present

    def handle_describe_click(self):
        for dataset, file_path in zip(self.datasets, self.file_paths):
            data_window = DataWindow(dataset, os.path.basename(file_path), self, self)
            self.data_windows.append(data_window)
            data_window.show()  # Use show() instead of exec_()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse Dataset")
        self.browse_button.clicked.connect(self.handle_browse_click)
        browse_layout.addWidget(self.browse_button)
        main_layout.addLayout(browse_layout)

        button_layout = QHBoxLayout()
        self.describe_button = QPushButton("Show Me The Data (Original)")
        self.describe_button.clicked.connect(self.handle_describe_click)
        self.describe_button.setEnabled(False)
        button_layout.addWidget(self.describe_button)

        main_layout.addLayout(button_layout)

        self.decision_tree_button = QPushButton("Decision Tree and Confusion Matrix")
        self.decision_tree_button.clicked.connect(self.handle_decision_tree_click)
        self.decision_tree_button.setVisible(False)
        main_layout.addWidget(self.decision_tree_button)

        self.setLayout(main_layout)

    def handle_browse_click(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Datasets (*.csv *.xls *.xlsx)")
        if file_dialog.exec_():
            try:
                selected_files = file_dialog.selectedFiles()
                self.file_paths = selected_files
                self.datasets = []
                for selected_file_path in selected_files:
                    if selected_file_path.endswith('.csv'):
                        data = pd.read_csv(selected_file_path)
                    elif selected_file_path.endswith(('.xls', '.xlsx')):
                        data = pd.read_excel(selected_file_path)
                    else:
                        raise ValueError("Unsupported file format")

                    self.datasets.append(data)
                self.describe_button.setEnabled(True)
                # self.decision_making_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load datasets: {str(e)}")

    def handle_describe_click(self):
        for dataset, file_path in zip(self.datasets, self.file_paths):
            data_window = DataWindow(dataset, os.path.basename(file_path), self, self)
            self.data_windows.append(data_window)
            data_window.show()  # Show all DataWindow instances

    def handle_data_description_click(self):
        dialog = DataDescriptionDialog(self.datasets, self.file_paths)
        dialog.exec_()

    def handle_prepare_data_click(self):
        if self.datasets:
            self.show_prepare_data_dialog()
        else:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first.")

    def handle_decision_tree_click(self):
        print("Decision tree visualization triggered in handle_decision_tree_click method")
        if self.X_train is None or self.y_train is None or self.feature_names is None:
            QMessageBox.warning(self, "Warning", "Please prepare the dataset first.")
            return
        show_decision_tree(self.X_train, self.y_train, self.feature_names)
        

    def handle_merge_data(self, selected_data, file_name):
        if self.merged_data is None:
            self.merged_data = selected_data
        else:
            self.merged_data = pd.concat([self.merged_data, selected_data], axis=1)

        new_window = DataWindow(self.merged_data, f"Merged from multiple files", self, merged_data=self.merged_data)
        self.data_windows.append(new_window)
        new_window.show()  # Change this from new_window.exec_() to new_window.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def show_prepare_data_dialog(self):
        feature_names = self.datasets[0].columns.tolist() if self.datasets else []
        dialog = PrepareDataDialog(self.datasets, feature_names, self, self)
        dialog.show()

    def handle_column_reordering(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Reorder Columns")
        dialog.setLabelText("Enter the new column order (comma-separated indices):")
        dialog.setTextEchoMode(QLineEdit.Normal)

        # Set a validator to accept only integers
        int_validator = QIntValidator(dialog)
        dialog.setTextValueValidator(int_validator)

        if dialog.exec_():
            column_order = dialog.textValue().split(",")
            dataset = self.dataset
            dataset.columns = [dataset.columns[int(idx)] for idx in column_order]

class NeuralNetworkResults(QDialog):
    def __init__(self, data, target_column, params, result, parent=None):
        super().__init__(parent)
        self.data = data
        self.target_column = target_column
        self.params = params
        self.result = result
        
        # Unpack the result dictionary
        self.model = result['model']
        self.X_test = result['X_test']
        self.y_test = result['y_test']
        self.preprocessor = result['preprocessor']
        self.is_regression = result['is_regression']
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Model Results Tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        self.create_model_results_content(model_layout)
        tab_widget.addTab(model_tab, "Model Results")

        # Hyperparameter Tuning Tab (if applicable)
        if self.result.get('all_results'):
            hyper_tab = QWidget()
            hyper_layout = QVBoxLayout(hyper_tab)
            self.create_hyperparameter_results_content(hyper_layout)
            tab_widget.addTab(hyper_tab, "Hyperparameter Tuning")

        self.setWindowTitle("Neural Network Results")
        self.resize(800, 600)

    def create_model_results_content(self, layout):
        # Display metrics
        metrics_text = QTextEdit()
        metrics_text.setReadOnly(True)
        html_content = "<h3>Model Metrics:</h3>"
        
        for col, metrics in self.result['metrics'].items():
            html_content += f"<h4>Metrics for {col}:</h4>"
            for metric_name, value in metrics.items():
                html_content += f"<b>{metric_name}:</b> {value:.4f}<br>"
        
        metrics_text.setHtml(html_content)
        layout.addWidget(metrics_text)

        # Display learning curves
        if 'history' in self.result:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(self.result['history'].history['loss'], label='Training Loss')
            ax.plot(self.result['history'].history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.set_title('Learning Curves')
            
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

        # Add buttons for LIME, SHAP, etc.
        button_layout = QHBoxLayout()
        
        lime_button = QPushButton("Show LIME Results")
        lime_button.clicked.connect(self.show_lime_results)
        button_layout.addWidget(lime_button)

        shap_button = QPushButton("Show SHAP Results")
        shap_button.clicked.connect(self.show_shap_results)
        button_layout.addWidget(shap_button)

        if not self.result['is_regression']:
            cm_button = QPushButton("Show Confusion Matrix")
            cm_button.clicked.connect(self.show_confusion_matrix)
            button_layout.addWidget(cm_button)

        layout.addLayout(button_layout)

    def create_hyperparameter_results_content(self, layout):
        results = self.result.get('all_results')
        if not results:
            layout.addWidget(QLabel("No hyperparameter tuning results available."))
            return

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        results_text = QTextEdit()
        results_text.setReadOnly(True)
        
        html_content = "<h3>Hyperparameter Tuning Results:</h3>"
        html_content += "<table border='1'><tr><th>Parameters</th><th>Mean Test Score</th><th>Std Test Score</th><th>Rank</th></tr>"
        
        for i, params in enumerate(results['params']):
            html_content += f"<tr><td>{params}</td>"
            html_content += f"<td>{results['mean_test_score'][i]:.4f}</td>"
            html_content += f"<td>{results['std_test_score'][i]:.4f}</td>"
            html_content += f"<td>{results['rank_test_score'][i]}</td></tr>"
        
        html_content += "</table>"
        
        if 'best_params_' in results:
            html_content += "<h4>Best Parameters:</h4>"
            html_content += f"<pre>{results['best_params_']}</pre>"
        
        results_text.setHtml(html_content)
        scroll_layout.addWidget(results_text)

        # Add a new section for individual hyperparameter accuracies
        accuracies_text = QTextEdit()
        accuracies_text.setReadOnly(True)
        
        html_content = "<h3>Accuracies for Each Hyperparameter Set:</h3>"
        html_content += "<table border='1'><tr><th>Parameters</th><th>Accuracy</th></tr>"
        
        for i, params in enumerate(results['params']):
            accuracy = results['mean_test_score'][i]
            html_content += f"<tr><td>{params}</td><td>{accuracy:.4f}</td></tr>"
        
        html_content += "</table>"
        
        accuracies_text.setHtml(html_content)
        scroll_layout.addWidget(accuracies_text)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

    def show_lime_results(self):
        lime_window = QDialog(self)
        lime_layout = QVBoxLayout()

        # Determine if it's a Keras model or a scikit-learn pipeline
        is_keras_model = isinstance(self.model, tf.keras.Model)

        # Convert X_test to numpy array if it's a DataFrame
        if isinstance(self.X_test, pd.DataFrame):
            X_test_array = self.X_test.values
        else:
            X_test_array = self.X_test

        # Determine if it's a regression or classification task
        if is_keras_model:
            is_regression = self.model.output_shape[-1] == 1
            num_classes = self.model.output_shape[-1]
        else:
            is_regression = self.is_regression
            num_classes = len(np.unique(self.y_test)) if not is_regression else 1

        feature_names = self.preprocessor.get_feature_names_out()
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test_array,
            feature_names=feature_names,
            class_names=[f'Class {i}' for i in range(num_classes)] if not is_regression else None,
            mode='regression' if is_regression else 'classification'
        )

        instance_idx = np.random.randint(0, X_test_array.shape[0])
        instance = X_test_array[instance_idx]

        def predict_fn(x):
            if is_keras_model:
                return self.model.predict(x)
            else:
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(x)
                else:
                    return self.model.predict(x)

        exp = explainer.explain_instance(
            instance, 
            predict_fn,
            num_features=10,
            num_samples=5000
        )

        # Create and display the LIME plot
        fig = exp.as_pyplot_figure()
        canvas = FigureCanvas(fig)
        lime_layout.addWidget(canvas)

        # Add textual explanation
        text_explanation = QTextEdit()
        text_explanation.setReadOnly(True)
        text_explanation.setHtml("<h3>LIME Explanation:</h3>")
        for feature, importance in exp.as_list():
            text_explanation.append(f"<b>{feature}:</b> {importance:.4f}<br>")
        lime_layout.addWidget(text_explanation)

        lime_window.setLayout(lime_layout)
        lime_window.setWindowTitle("LIME Results")
        lime_window.resize(800, 600)
        lime_window.show()

    def show_shap_results(self):
        shap_window = QDialog(self)
        shap_layout = QVBoxLayout()

        try:
            # Convert to DataFrame if necessary
            if not isinstance(self.X_test, pd.DataFrame):
                X_test_df = pd.DataFrame(self.X_test, columns=self.preprocessor.get_feature_names_out())
            else:
                X_test_df = self.X_test

            # Ensure all features are numeric and convert to float64
            X_test_numeric = X_test_df.select_dtypes(include=[np.number]).astype(np.float64)

            # Handle missing values
            X_test_numeric = X_test_numeric.fillna(X_test_numeric.mean())

            # Create a function that mimics the model's predict method
            def model_predict(x):
                predictions = self.model.predict(x)
                # If predictions is a 2D array, return the first column
                if predictions.ndim > 1:
                    return predictions[:, 0]
                return predictions

            # Use KernelExplainer
            explainer = shap.KernelExplainer(model_predict, X_test_numeric[:100])

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_numeric[:100])

            # Ensure shap_values is 2D
            if shap_values.ndim == 3:
                shap_values = shap_values[0]

            # Calculate mean absolute SHAP values for each feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame(list(zip(X_test_numeric.columns, mean_shap)), 
                                            columns=['feature', 'importance'])
            feature_importance = feature_importance.sort_values('importance', ascending=True)

            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title("SHAP Feature Importance")
            plt.xlabel("Mean |SHAP Value|")
            plt.tight_layout()

            canvas = FigureCanvas(plt.gcf())
            shap_layout.addWidget(canvas)

            # Add textual explanation
            text_explanation = QTextEdit()
            text_explanation.setReadOnly(True)
            text_explanation.setHtml("<h3>SHAP Explanation:</h3>")

            for _, row in feature_importance.iterrows():
                text_explanation.append(f"<b>{row['feature']}:</b> {row['importance']:.4f}<br>")

            shap_layout.addWidget(text_explanation)

        except Exception as e:
            error_message = QLabel(f"Error in SHAP computation: {str(e)}")
            shap_layout.addWidget(error_message)
            print(f"Detailed error: {e}")

        shap_window.setLayout(shap_layout)
        shap_window.setWindowTitle("SHAP Results")
        shap_window.resize(800, 600)
        shap_window.show()

    def show_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        confusion_matrix_window = QDialog(self)
        confusion_matrix_window.setWindowTitle("Confusion Matrix")
        layout = QVBoxLayout()

        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        confusion_matrix_window.setLayout(layout)
        confusion_matrix_window.resize(800, 600)
        confusion_matrix_window.show()

    def format_classification_report(self, report):
        html = "<style>table {border-collapse: collapse;} th, td {border: 1px solid black; padding: 8px;}</style>"
        html += "<h2>Classification Report</h2>"
        html += "<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>"
        
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            html += f"<tr><td>{class_name}</td>"
            html += f"<td>{metrics['precision']:.2f}</td>"
            html += f"<td>{metrics['recall']:.2f}</td>"
            html += f"<td>{metrics['f1-score']:.2f}</td>"
            html += f"<td>{metrics['support']}</td></tr>"
        
        html += f"<tr><td colspan='5'><b>Accuracy: {report['accuracy']:.2f}</b></td></tr>"
        html += "</table>"
        return html
    
class DialogInfo:
    def __init__(self, dialog_type, parameters):
        self.dialog_type = dialog_type
        self.parameters = parameters
        self.instance_count = 1  # Add this line

    def __eq__(self, other):
        return (self.dialog_type == other.dialog_type and 
                self.parameters == other.parameters)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())