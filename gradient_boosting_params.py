from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QCheckBox, QListWidget, QAbstractItemView)
from PyQt5.QtCore import Qt

class GradientBoostingParamsDialog(QDialog):
    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        self.setWindowTitle("Gradient Boosting Machine Parameters")
        self.columns = columns or []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Column selection
        column_selection_layout = QHBoxLayout()

        # Training columns
        train_layout = QVBoxLayout()
        train_layout.addWidget(QLabel("Training Columns:"))
        self.train_list = QListWidget()
        self.train_list.addItems(self.columns)
        self.train_list.setSelectionMode(QAbstractItemView.MultiSelection)
        train_layout.addWidget(self.train_list)
        column_selection_layout.addLayout(train_layout)

        # Testing column
        test_layout = QVBoxLayout()
        test_layout.addWidget(QLabel("Testing Column:"))
        self.test_list = QListWidget()
        self.test_list.addItems(self.columns)
        self.test_list.setSelectionMode(QAbstractItemView.SingleSelection)
        test_layout.addWidget(self.test_list)
        column_selection_layout.addLayout(test_layout)

        layout.addLayout(column_selection_layout)

        # Existing parameters
        params_layout = QVBoxLayout()

        # Test Size
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(QLabel("Test Size:"))
        self.test_size_edit = QLineEdit()
        self.test_size_edit.setText("0.2")  # Default value
        test_size_layout.addWidget(self.test_size_edit)
        params_layout.addLayout(test_size_layout)

        # Number of Estimators
        n_estimators_layout = QHBoxLayout()
        n_estimators_layout.addWidget(QLabel("Number of Estimators:"))
        self.n_estimators_edit = QLineEdit()
        self.n_estimators_edit.setText("100")  # Default value
        n_estimators_layout.addWidget(self.n_estimators_edit)
        params_layout.addLayout(n_estimators_layout)

        # Learning Rate
        learning_rate_layout = QHBoxLayout()
        learning_rate_layout.addWidget(QLabel("Learning Rate:"))
        self.learning_rate_edit = QLineEdit()
        self.learning_rate_edit.setText("0.1")  # Default value
        learning_rate_layout.addWidget(self.learning_rate_edit)
        params_layout.addLayout(learning_rate_layout)

        # Max Depth
        max_depth_layout = QHBoxLayout()
        max_depth_layout.addWidget(QLabel("Max Depth:"))
        self.max_depth_edit = QLineEdit()
        self.max_depth_edit.setText("3")  # Default value
        max_depth_layout.addWidget(self.max_depth_edit)
        params_layout.addLayout(max_depth_layout)

        # Show Plots
        self.show_plots_checkbox = QCheckBox("Show Plots")
        self.show_plots_checkbox.setChecked(True)  # Default value
        params_layout.addWidget(self.show_plots_checkbox)

        layout.addLayout(params_layout)

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

    def get_params(self):
        return {
            "train_columns": [item.text() for item in self.train_list.selectedItems()],
            "test_column": self.test_list.currentItem().text() if self.test_list.currentItem() else None,
            "test_size": float(self.test_size_edit.text()),
            "n_estimators": int(self.n_estimators_edit.text()),
            "learning_rate": float(self.learning_rate_edit.text()),
            "max_depth": int(self.max_depth_edit.text()),
            "show_plots": self.show_plots_checkbox.isChecked()
        }