import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import seaborn as sns

def prepare_and_train_knn(data, params):
    # Extract parameters
    test_size = params.get('test_size', 0.2)
    n_neighbors = params.get('n_neighbors', 5)
    weights = params.get('weights', 'uniform')
    train_columns = params.get('train_columns', [])
    target_column = params.get('test_column')

    # Separate features and target
    X = data[train_columns]
    y = data[target_column]

    # Determine if it's a classification or regression problem
    if y.dtype == 'object' or y.nunique() < 10:
        task = 'classification'
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
    else:
        task = 'regression'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create and train the KNN model
    if task == 'classification':
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    else:
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Evaluate the model
    if task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")

        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.show()

    # Feature importance (using permutation importance)
    perm_importance = permutation_importance(model, X_test_processed, y_test, n_repeats=10, random_state=42)
    feature_names = numeric_features.tolist() + [f"{feature}_{cat}" for feature in categorical_features for cat in preprocessor.named_transformers_['cat'].categories_[list(categorical_features).index(feature)][1:]]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot the effect of K on model performance
    k_values = range(1, 31)
    train_scores = []
    test_scores = []

    for k in k_values:
        if task == 'classification':
            knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        else:
            knn = KNeighborsRegressor(n_neighbors=k, weights=weights)
        
        knn.fit(X_train_processed, y_train)
        train_scores.append(knn.score(X_train_processed, y_train))
        test_scores.append(knn.score(X_test_processed, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_scores, label='Training Score')
    plt.plot(k_values, test_scores, label='Testing Score')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Model Score')
    plt.title('Effect of K on Model Performance')
    plt.legend()
    plt.show()

    return model, X_train_processed, X_test_processed, y_train, y_test, preprocessor

# Example usage:
# Assuming 'data' is your pandas DataFrame and 'target_column' is the name of your target variable
# model, X_test, y_test, preprocessor = prepare_and_train_knn(data, 'target_column')
