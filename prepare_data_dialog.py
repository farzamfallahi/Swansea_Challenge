from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView

class PrepareDataDialog(QDialog):
    def __init__(self, X_train, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepared Data")
        self.resize(800, 600)
        self.setup_ui(X_train, feature_names)

    def setup_ui(self, X_train, feature_names):
        layout = QVBoxLayout(self)
        table = QTableWidget()
        table.setRowCount(len(X_train))
        table.setColumnCount(len(feature_names))
        table.setHorizontalHeaderLabels(feature_names)
        for row in range(len(X_train)):
            for col in range(len(feature_names)):
                item = QTableWidgetItem(str(X_train[row, col]))
                table.setItem(row, col, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(table)
        self.setLayout(layout)