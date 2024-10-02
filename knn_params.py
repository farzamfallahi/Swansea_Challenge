from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QListWidget, QAbstractItemView,
                             QSlider, QLineEdit)
from PyQt5.QtCore import Qt

class KNNParamsDialog(QDialog):
    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        self.setWindowTitle("K-Nearest Neighbors Parameters")
        self.columns = columns or []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Number of neighbors
        self.n_neighbors_slider, self.n_neighbors_display = self.create_param_control(
            "Number of Neighbors", 1, 20, 5, 1, layout)

        # Weights
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(QLabel("Weights:"))
        self.weights_combo = QComboBox()
        self.weights_combo.addItems(["uniform", "distance"])
        weights_layout.addWidget(self.weights_combo)
        layout.addLayout(weights_layout)

        # Test Size
        self.test_size_slider, self.test_size_display = self.create_param_control(
            "Test Size", 0.1, 0.5, 0.2, 0.1, layout, is_float=True)

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
            "n_neighbors": self.n_neighbors_slider.value(),
            "weights": self.weights_combo.currentText(),
            "test_size": self.test_size_slider.value() * 0.1,
            "train_columns": [item.text() for item in self.train_list.selectedItems()],
            "test_column": self.test_list.selectedItems()[0].text() if self.test_list.selectedItems() else None
        }