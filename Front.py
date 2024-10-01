import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QLabel)
from PyQt5.QtCore import Qt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.df = None
        self.column_names = None  # Store column names here
        
    def initUI(self):
        self.setWindowTitle('Data Processing Application')
        self.layout = QVBoxLayout()
        
        self.browseButton = QPushButton('Browse Dataset')
        self.browseButton.clicked.connect(self.browseDataset)
        self.layout.addWidget(self.browseButton)
        
        self.processButton = QPushButton('Process Dataset')
        self.processButton.clicked.connect(self.processDataset)
        self.layout.addWidget(self.processButton)
        
        self.resultLabel = QLabel("Results will be shown here.")
        self.layout.addWidget(self.resultLabel)
        
        self.table = QTableWidget()
        self.layout.addWidget(self.table)
        
        self.setLayout(self.layout)
        
    def browseDataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            self.df = pd.read_csv(filePath)
            self.resultLabel.setText(f"Dataset loaded: {filePath}")
    
    def processDataset(self):
        if self.df is not None:
            X_train, X_test, y_train, y_test, self.column_names = self.prepare_data(self.df)
            self.showResults(X_train, X_test, y_train, y_test, self.column_names)
        else:
            self.resultLabel.setText("Please load a dataset first.")
    
    def prepare_data(self, df):
        # 2. Exploratory Data Analysis (EDA)
        print("Initial Shape:", df.shape)
        print("Initial Data Types:\n", df.dtypes)
        print("Initial Data Description:\n", df.describe())
        print("Checking for missing values:\n", df.isnull().sum())
        
        # 3. Handle Missing Values (if any)
        df = df.fillna(df.mean())  # Fill missing values with the mean
        
        # 4. Feature Engineering (if needed)
        df = pd.get_dummies(df, drop_first=True)
        
        # 5. Split dataset into features and target
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        
        # Store column names for display
        column_names = X.columns
        
        # 6. Normalize/Standardize Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 7. Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Print the shapes of the final datasets
        print("Training Features Shape:", X_train.shape)
        print("Training Labels Shape:", y_train.shape)
        print("Testing Features Shape:", X_test.shape)
        print("Testing Labels Shape:", y_test.shape)
        
        return X_train, X_test, y_train, y_test, column_names
    
    def showResults(self, X_train, X_test, y_train, y_test, column_names):
        # Convert y_train to a NumPy array
        y_train_array = y_train.to_numpy()
        
        # Show data in a table
        self.table.setRowCount(10)  # Example to show 10 rows
        self.table.setColumnCount(X_train.shape[1] + 1)
        self.table.setHorizontalHeaderLabels(['Outcome'] + list(column_names))
        
        for i in range(10):
            self.table.setItem(i, 0, QTableWidgetItem(str(y_train_array[i])))
            for j in range(X_train.shape[1]):
                self.table.setItem(i, j + 1, QTableWidgetItem(str(X_train[i, j])))
        
        self.resultLabel.setText("Data processed and displayed below.")

def main():
    app = QApplication(sys.argv)
    window = DataApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
