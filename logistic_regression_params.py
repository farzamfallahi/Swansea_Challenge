from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QCheckBox, QListWidget, QAbstractItemView,
                             QSlider, QLineEdit)
from PyQt5.QtCore import Qt

class LogisticRegressionParamsDialog(QDialog):
    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        self.setWindowTitle("Logistic Regression Parameters")
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

        # Testing columns (allows multiple selection)
        test_layout = QVBoxLayout()
        test_layout.addWidget(QLabel("Testing Column(s):"))
        self.test_list = QListWidget()
        self.test_list.addItems(self.columns)
        self.test_list.setSelectionMode(QAbstractItemView.MultiSelection)
        test_layout.addWidget(self.test_list)
        column_selection_layout.addLayout(test_layout)

        layout.addLayout(column_selection_layout)

        # Parameters with sliders
        params_layout = QVBoxLayout()

        # Test Size
        self.test_size_slider, self.test_size_display = self.create_param_control(
            "Test Size", 0.1, 0.5, 0.2, 0.1, params_layout, is_float=True)

        # C (Inverse of regularization strength)
        self.c_slider, self.c_display = self.create_param_control(
            "C (Inverse of regularization strength)", 0.1, 10.0, 1.0, 0.1, params_layout, is_float=True)

        # Max Iterations
        self.max_iter_slider, self.max_iter_display = self.create_param_control(
            "Max Iterations", 100, 2000, 1000, 100, params_layout)

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
            "train_columns": [item.text() for item in self.train_list.selectedItems()],
            "test_columns": [item.text() for item in self.test_list.selectedItems()],
            "test_size": self.test_size_slider.value() * 0.1,
            "C": self.c_slider.value() * 0.1,
            "max_iter": self.max_iter_slider.value(),
            "show_plots": self.show_plots_checkbox.isChecked()
        }