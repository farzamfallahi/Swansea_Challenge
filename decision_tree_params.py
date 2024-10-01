from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QListWidget,
                             QAbstractItemView, QSlider, QLineEdit, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt

class DecisionTreeParamsDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Training columns selection
        train_layout = QVBoxLayout()
        train_label = QLabel("Select training columns:")
        self.train_list = QListWidget()
        self.train_list.addItems(self.columns)
        self.train_list.setSelectionMode(QAbstractItemView.MultiSelection)
        train_layout.addWidget(train_label)
        train_layout.addWidget(self.train_list)
        layout.addLayout(train_layout)

        # Test column selection
        test_layout = QHBoxLayout()
        test_label = QLabel("Select test column:")
        self.test_combo = QComboBox()
        self.test_combo.addItems(self.columns)
        test_layout.addWidget(test_label)
        test_layout.addWidget(self.test_combo)
        layout.addLayout(test_layout)

        # Hyperparameters
        params_layout = QVBoxLayout()

        # Max Depth
        self.depth_slider, self.depth_display = self.create_param_control(
            "Max Depth", 1, 30, 5, params_layout)

        # Min Samples Split
        self.split_slider, self.split_display = self.create_param_control(
            "Min Samples Split", 2, 20, 2, params_layout)

        # Min Samples Leaf
        self.leaf_slider, self.leaf_display = self.create_param_control(
            "Min Samples Leaf", 1, 20, 1, params_layout)

        layout.addLayout(params_layout)

        # Add Grid Search Option
        self.grid_search_checkbox = QCheckBox("Use Grid Search")
        layout.addWidget(self.grid_search_checkbox)

        # Grid Search Parameters
        self.grid_search_params = {}
        params_to_tune = ['max_depth', 'min_samples_split', 'min_samples_leaf']
        for param in params_to_tune:
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param} values:"))
            self.grid_search_params[param] = QLineEdit()
            self.grid_search_params[param].setPlaceholderText("Enter comma-separated values")
            param_layout.addWidget(self.grid_search_params[param])
            layout.addLayout(param_layout)

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
        self.setWindowTitle("Decision Tree Parameters")

    def create_param_control(self, label, min_val, max_val, default, parent_layout):
        param_layout = QVBoxLayout()
        param_label = QLabel(f"{label} ({min_val}-{max_val}):")
        param_layout.addWidget(param_label)

        input_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.setValue(default)
        slider.setTickPosition(QSlider.TicksBelow)
        
        display = QLineEdit()
        display.setFixedWidth(30)
        display.setText(str(default))
        
        input_layout.addWidget(slider)
        input_layout.addWidget(display)
        
        param_layout.addLayout(input_layout)
        parent_layout.addLayout(param_layout)

        # Connect slider and line edit
        slider.valueChanged.connect(lambda value: self.update_display(display, value))
        display.editingFinished.connect(lambda: self.update_slider(slider, display, min_val, max_val))

        return slider, display

    def update_display(self, display, value):
        display.setText(str(value))

    def update_slider(self, slider, display, min_val, max_val):
        try:
            value = int(display.text())
            if min_val <= value <= max_val:
                slider.setValue(value)
            else:
                display.setText(str(slider.value()))
        except ValueError:
            display.setText(str(slider.value()))

    def get_params(self):
        params = {
            "train_columns": [item.text() for item in self.train_list.selectedItems()],
            "test_column": self.test_combo.currentText(),
            "max_depth": self.depth_slider.value(),
            "min_samples_split": self.split_slider.value(),
            "min_samples_leaf": self.leaf_slider.value(),
            "use_grid_search": self.grid_search_checkbox.isChecked(),
        }
        
        if params["use_grid_search"]:
            params["grid_search_params"] = {}
            error_messages = []
            
            for param, line_edit in self.grid_search_params.items():
                values = line_edit.text().strip()
                if values:
                    try:
                        # Convert to appropriate type (int for these parameters)
                        param_values = [int(v.strip()) for v in values.split(',') if v.strip()]
                        if not param_values:
                            error_messages.append(f"No valid values provided for {param}")
                        else:
                            params["grid_search_params"][param] = param_values
                    except ValueError:
                        error_messages.append(f"Invalid input for {param}. Please enter comma-separated integers.")
                else:
                    error_messages.append(f"No values provided for {param}")
            
            if error_messages:
                error_message = "The following errors occurred:\n" + "\n".join(error_messages)
                QMessageBox.warning(self, "Invalid Input", error_message)
                return None
            
            if not params["grid_search_params"]:
                QMessageBox.warning(self, "No Parameters", "No valid grid search parameters provided.")
                return None
        
        return params

    def accept(self):
        if self.get_params() is not None:
            super().accept()