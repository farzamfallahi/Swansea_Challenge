from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView

class DataDescriptionDialog(QDialog):
    def __init__(self, description, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initial Data Description")
        self.resize(800, 600)
        self.setup_ui(description)

    def setup_ui(self, description):
        layout = QVBoxLayout(self)
        table = QTableWidget()
        table.setRowCount(len(description))
        table.setColumnCount(len(description.columns))
        table.setHorizontalHeaderLabels(description.columns)
        table.setVerticalHeaderLabels(description.index)

        for row in range(len(description)):
            for col in range(len(description.columns)):
                item = QTableWidgetItem(str(description.iat[row, col]))
                table.setItem(row, col, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(table)
        self.setLayout(layout)
