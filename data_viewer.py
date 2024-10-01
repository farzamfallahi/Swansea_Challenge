from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView

class DataViewerDialog(QDialog):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.setWindowTitle("Data Viewer")
        self.resize(800, 600)  # Set initial size but allow resizing
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        table = QTableWidget()
        table.setRowCount(len(self.data))
        table.setColumnCount(len(self.data.columns))
        table.setHorizontalHeaderLabels(self.data.columns)

        for row in range(len(self.data)):
            for column in range(len(self.data.columns)):
                item = QTableWidgetItem(str(self.data.iat[row, column]))
                table.setItem(row, column, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(table)

        self.setLayout(layout)
