# preparation.py
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

def import_dataset(file_path):
    """Step 1: Importing Dataset"""
    with open(file_path, 'r') as f:
        data = f.read()
    df = pd.read_csv(io.StringIO(data))
    return df

def exploratory_data_analysis(df):
    """Step 2: Exploratory Data Analysis (EDA)"""
    print("Initial Shape:", df.shape)
    print("Initial Data Types:\n", df.dtypes)
    print("Initial Data Description:\n", df.describe())
    missing_values = df.isnull().sum()
    print("Checking for missing values:\n", missing_values)
   
    # Calculate percentage of missing values
    total_cells = df.size
    if total_cells == 0:
        missing_percentage = 0
    else:
        total_missing = missing_values.sum()
        missing_percentage = (total_missing / total_cells) * 100
    print("Percentage of Missing Values: {:.2f}%".format(missing_percentage))
   
    return missing_values

def mean_imputation(df):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def median_imputation(df):
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def mode_imputation(df):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def knn_imputation(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def multiple_imputation(df, n_imputations=5):
    imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=n_imputations, random_state=42)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def regression_imputation(df):
    imputer = IterativeImputer(random_state=42)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def decision_tree_imputation(df):
    imputer = IterativeImputer(estimator=DecisionTreeRegressor(max_features='sqrt', random_state=42), random_state=42)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def flag_missing_values(df):
    for column in df.columns:
        if df[column].isnull().any():
            df[f'{column}_missing'] = df[column].isnull().astype(int)
    return df

def handle_missing_values(df, method='mean'):
    """Step 3: Handle Missing Values"""
    numeric_df = df.select_dtypes(include=['number'])
    non_numeric_df = df.select_dtypes(exclude=['number'])
    
    if method == 'mean':
        imputed_df = mean_imputation(numeric_df)
    elif method == 'median':
        imputed_df = median_imputation(numeric_df)
    elif method == 'mode':
        imputed_df = mode_imputation(numeric_df)
    elif method == 'knn':
        imputed_df = knn_imputation(numeric_df)
    elif method == 'multiple':
        imputed_df = multiple_imputation(numeric_df)
    elif method == 'regression':
        imputed_df = regression_imputation(numeric_df)
    elif method == 'decision_tree':
        imputed_df = decision_tree_imputation(numeric_df)
    elif method == 'flag':
        imputed_df = flag_missing_values(numeric_df)
    else:
        raise ValueError("Invalid imputation method")
    
    # Handle non-numeric columns (e.g., with mode imputation)
    if not non_numeric_df.empty:
        non_numeric_imputed = mode_imputation(non_numeric_df)
        imputed_df = pd.concat([imputed_df, non_numeric_imputed], axis=1)
    
    return imputed_df

def feature_engineering(df):
    """Step 4: Feature Engineering (if needed)"""
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_features_and_target(df, target_column='Outcome'):
    """Step 5: Split dataset into features and target"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def normalize_features(X):
    """Step 6: Normalize/Standardize Features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_data(X, y, test_size=0.2, random_state=42):
    """Step 7: Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def prepare_data(file_path, imputation_method='mean', target_column='Outcome'):
    df = import_dataset(file_path)
    missing_values = exploratory_data_analysis(df)
    df = handle_missing_values(df, method=imputation_method)
    df = feature_engineering(df)
    X, y = split_features_and_target(df, target_column)
    X_scaled = normalize_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Print the shapes of the final datasets
    print("Training Features Shape:", X_train.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Testing Labels Shape:", y_test.shape)
    
    return (X_train, X_test, y_train, y_test), X.columns.tolist(), missing_values, df

def min_max_scaling(X):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def z_score_standardization(X):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def robust_scaling(X):
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def max_absolute_scaling(X):
    scaler = MaxAbsScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def normalize_features(X, method='z_score'):
    """Step 6: Normalize/Standardize Features"""
    if method == 'min_max':
        X_scaled = min_max_scaling(X)
    elif method == 'z_score':
        X_scaled = z_score_standardization(X)
    elif method == 'robust':
        X_scaled = robust_scaling(X)
    elif method == 'max_abs':
        X_scaled = max_absolute_scaling(X)
    else:
        raise ValueError("Invalid normalization method")
    return X_scaled

def split_data(X, y, test_size=0.2, random_state=42):
    """Step 7: Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def prepare_data(file_path, imputation_method='mean', normalization_method='z_score', target_column='Outcome'):
    df = import_dataset(file_path)
    missing_values = exploratory_data_analysis(df)
    df = handle_missing_values(df, method=imputation_method)
    df = feature_engineering(df)
    X, y = split_features_and_target(df, target_column)
    X_scaled = normalize_features(X, method=normalization_method)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Print the shapes of the final datasets
    print("Training Features Shape:", X_train.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Testing Labels Shape:", y_test.shape)
    
    return (X_train, X_test, y_train, y_test), X.columns.tolist(), missing_values, df

def handle_outliers(df, method='zscore', threshold=3):
    """Step 3.5: Handle Outliers"""
    numeric_df = df.select_dtypes(include=['number'])
    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(numeric_df))
        df_without_outliers = df[(z_scores < threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        df_without_outliers = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif method == 'winsorize':
        df_without_outliers = df.copy()
        for column in numeric_df.columns:
            lower_bound, upper_bound = np.percentile(df[column], [2.5, 97.5])
            df_without_outliers[column] = np.clip(df[column], lower_bound, upper_bound)
    else:
        raise ValueError("Invalid outlier handling method")
    
    print(f"Rows removed: {len(df) - len(df_without_outliers)}")
    return df_without_outliers

def prepare_data(file_path, imputation_method='mean', normalization_method='z_score', outlier_method='zscore', target_column='Outcome'):
    df = import_dataset(file_path)
    missing_values = exploratory_data_analysis(df)
    df = handle_missing_values(df, method=imputation_method)
    df = handle_outliers(df, method=outlier_method)  # New step
    df = feature_engineering(df)
    X, y = split_features_and_target(df, target_column)
    X_scaled = normalize_features(X, method=normalization_method)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Print the shapes of the final datasets
    print("Training Features Shape:", X_train.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Testing Labels Shape:", y_test.shape)
    
    return (X_train, X_test, y_train, y_test), X.columns.tolist(), missing_values, df