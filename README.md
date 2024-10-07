# Swansea_Challenge
A software which is the showcase of my skills
Certainly! Here's the updated README.md with your name and GitHub repository:

# XAI Data Analysis and Machine Learning Application

This project is a comprehensive GUI application for data analysis and machine learning, featuring data preparation, visualization, model implementation, and explainability techniques.

## Features

- **Data Handling**: 
  - Load and view datasets (CSV, XLS, XLSX)
  - Data preprocessing and cleaning
  - Handling missing values and outliers
  - Feature engineering and normalization

- **Exploratory Data Analysis (EDA)**:
  - Data description and statistics
  - Various plotting options (histograms, scatter plots)

- **Machine Learning Models**:
  - Decision Trees
  - Neural Networks
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - Gradient Boosting Machines (GBM)
  - Logistic Regression

- **Model Explainability**:
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (SHapley Additive exPlanations)
  - ELI5 (Explain Like I'm 5)
  - Confusion Matrix visualization

- **Time Series Analysis**:
  - LIME explanations for time series data

- **Hyperparameter Tuning**:
  - Grid search and randomized search options for various models

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/farzamfallahi/Swansea_Challenge.git
   cd Swansea_Challenge
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:
   ```
   python Front.py
   ```

2. Use the GUI to load your dataset, perform data preparation, and choose your desired analysis or machine learning model.

3. Explore the results and explanations provided by the various XAI techniques.

## Dependencies

- PyQt5
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- lime
- shap
- eli5

## File Structure

- `Front.py`: Main application GUI
- `data_viewer.py`: Data viewing dialog
- `decision_tree.py`: Decision tree implementation and visualization
- `Neural_Network.py`: Neural network model implementation
- `K_Nearest_Neighbors.py`: KNN algorithm implementation
- `Support_Vector_Machines.py`: SVM model implementation
- `Gradient_Boosting_Machines.py`: GBM implementation
- `Lime.py`: LIME analysis implementation
- `Shap.py`: SHAP analysis implementation
- `ELI5.py`: ELI5 explanations
- `preparation.py`: Data preparation and preprocessing functions
- `plotting_functions.py`: Various plotting utilities
- `confusion_matrix.py`: Custom confusion matrix implementation

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

FARZAM FALLAHY

Project Link: [https://github.com/farzamfallahi/Swansea_Challenge](https://github.com/farzamfallahi/Swansea_Challenge)