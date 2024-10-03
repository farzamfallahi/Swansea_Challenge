from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QAbstractItemView,
                             QSlider, QLineEdit, QTableWidget, QTableWidgetItem,
                             QComboBox, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt

class NeuralNetworkParamsDialog(QDialog):
    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        self.setWindowTitle("Neural Network Parameters")
        self.columns = columns or []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Add sliders for all numerical parameters
        self.test_size_slider, self.test_size_display = self.create_param_control(
            "Test Size", 0.1, 0.5, 0.2, 0.1, layout, is_float=True)

        self.epochs_slider, self.epochs_display = self.create_param_control(
            "Epochs", 10, 1000, 100, 10, layout)

        self.batch_size_slider, self.batch_size_display = self.create_param_control(
            "Batch Size", 1, 128, 32, 1, layout)

        self.learning_rate_slider, self.learning_rate_display = self.create_param_control(
            "Learning Rate", 0.0001, 0.1, 0.001, 0.0001, layout, is_float=True)

        # Layer configuration table
        layer_layout = QVBoxLayout()
        layer_layout.addWidget(QLabel("Layer Configuration:"))
        self.layer_table = QTableWidget(1, 3)
        self.layer_table.setHorizontalHeaderLabels(["Neurons", "Inputs", ""])
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        layer_layout.addWidget(self.layer_table)

        # Initialize the first layer
        self.init_layer_row(0)

        # Add and remove layer buttons
        button_layout = QHBoxLayout()
        add_layer_button = QPushButton("Add Layer")
        add_layer_button.clicked.connect(self.add_layer)
        button_layout.addWidget(add_layer_button)

        remove_layer_button = QPushButton("Remove Layer")
        remove_layer_button.clicked.connect(self.remove_layer)
        button_layout.addWidget(remove_layer_button)

        layer_layout.addLayout(button_layout)
        layout.addLayout(layer_layout)

        # Add column selection for training
        train_layout = QVBoxLayout()
        train_layout.addWidget(QLabel("Select Training Columns:"))
        self.train_list = QListWidget()
        self.train_list.addItems(self.columns)
        self.train_list.setSelectionMode(QAbstractItemView.MultiSelection)
        train_layout.addWidget(self.train_list)
        layout.addLayout(train_layout)

        # Add column selection for testing (multiple selection)
        test_layout = QVBoxLayout()
        test_layout.addWidget(QLabel("Select Testing Columns:"))
        self.test_list = QListWidget()
        self.test_list.addItems(self.columns)
        self.test_list.setSelectionMode(QAbstractItemView.MultiSelection)
        test_layout.addWidget(self.test_list)
        layout.addLayout(test_layout)

        # Add optimizer selection
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        optimizer_layout.addWidget(self.optimizer_combo)
        layout.addLayout(optimizer_layout)

        # Add Grid Search option
        self.grid_search_checkbox = QCheckBox("Use Grid Search")
        layout.addWidget(self.grid_search_checkbox)

        # Grid Search Parameters
        self.grid_search_params = {}
        params_to_tune = ['learning_rate', 'batch_size', 'epochs']
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
        display.setText(f"{value:.4f}" if isinstance(value, float) else str(value))

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

    def init_layer_row(self, row):
        # Neuron selection
        neuron_combo = QComboBox()
        neuron_combo.addItems(["32", "64"])
        self.layer_table.setCellWidget(row, 0, neuron_combo)

        # Activation function selection
        activation_combo = QComboBox()
        activation_combo.addItems(["relu", "sigmoid", "tanh"])
        self.layer_table.setCellWidget(row, 1, activation_combo)

        # Info button
        info_button = QPushButton("?")
        info_button.clicked.connect(self.show_activation_info)
        self.layer_table.setCellWidget(row, 2, info_button)

    def add_layer(self):
        row_count = self.layer_table.rowCount()
        self.layer_table.insertRow(row_count)
        self.init_layer_row(row_count)

    def remove_layer(self):
        if self.layer_table.rowCount() > 1:
            self.layer_table.removeRow(self.layer_table.rowCount() - 1)

    def show_input_info(self):
        QMessageBox.information(self, "Input Specification",
                                "Enter 'X' to use all inputs from the previous layer, "
                                "or enter a specific number of inputs.")
        
    def show_activation_info(self):
        QMessageBox.information(self, "Activation Function",
                                "Choose an activation function for this layer:\n"
                                "- ReLU: Rectified Linear Unit, good for hidden layers\n"
                                "- Sigmoid: Useful for binary classification output\n"
                                "- Tanh: Similar to sigmoid, but zero-centered")

    def get_params(self):
        params = {
            "test_size": self.test_size_slider.value() * 0.1,
            "epochs": self.epochs_slider.value(),
            "batch_size": self.batch_size_slider.value(),
            "learning_rate": self.learning_rate_slider.value() * 0.0001,
            "train_columns": [item.text() for item in self.train_list.selectedItems()],
            "test_columns": [item.text() for item in self.test_list.selectedItems()],
            "layers": [
                (int(self.layer_table.cellWidget(row, 0).currentText()), 
                self.layer_table.cellWidget(row, 1).currentText())
                for row in range(self.layer_table.rowCount())
            ],
            "optimizer": self.optimizer_combo.currentText(),
            "use_grid_search": self.grid_search_checkbox.isChecked(),
        }
        
        if params["use_grid_search"]:
            params["grid_search_params"] = {}
            for param, line_edit in self.grid_search_params.items():
                values = line_edit.text().strip()
                if values:
                    if param in ['learning_rate']:
                        params["grid_search_params"][param] = [float(v.strip()) for v in values.split(',') if v.strip()]
                    else:
                        params["grid_search_params"][param] = [int(v.strip()) for v in values.split(',') if v.strip()]
                else:
                    QMessageBox.warning(self, "Invalid Input", f"No values provided for {param}")
                    return None

        return params
    
    