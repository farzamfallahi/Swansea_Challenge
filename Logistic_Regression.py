import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def prepare_and_train_logistic_regression(X, y, params):
    # Extract parameters
    test_size = params.get('test_size', 0.2)
    C = params.get('C', 1.0)
    max_iter = params.get('max_iter', 1000)
    show_plots = params.get('show_plots', True)

    # Handle NaN values in the target variable
    nan_mask = y.notnull()
    X = X[nan_mask]
    y = y[nan_mask]

    # Convert y to numeric if it's not already
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        # If y is already numeric, ensure it starts from 0
        y = y - y.min()

    # Check if y is binary or multiclass
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("The target variable must have at least two unique classes.")

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_features),
            ('cat', SimpleImputer(strategy='constant', fill_value='missing'), categorical_features)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns=numeric_features.tolist() + categorical_features.tolist())

    # Check if stratification is possible
    if len(unique_classes) > 1 and np.min(np.bincount(y)) >= 2:
        stratify = y
    else:
        print("Warning: Stratification is not possible. Using random split.")
        stratify = None

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42, stratify=stratify)

    # Create the final preprocessor
    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    # Fit and transform the training data
    X_train_processed = final_preprocessor.fit_transform(X_train)
    X_test_processed = final_preprocessor.transform(X_test)

    # Create and train the Logistic Regression model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42, multi_class='ovr', class_weight='balanced')
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))


    if show_plots:
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        pass

    # Feature importance (using permutation importance)
    perm_importance = permutation_importance(model, X_test_processed, y_test, n_repeats=10, random_state=42)
    
    # Get feature names after preprocessing
    feature_names = numeric_features.tolist()
    for feature in categorical_features:
        unique_cats = X[feature].unique()
        unique_cats = [cat for cat in unique_cats if cat != 'missing' and not pd.isna(cat)]
        if len(unique_cats) > 1:
            feature_names.extend([f"{feature}_{cat}" for cat in unique_cats[1:]])  # Exclude first category due to drop='first'
        elif len(unique_cats) == 1:
            feature_names.append(feature)  # Add the feature name as is if there's only one unique category
                
    # Ensure feature_names matches the number of features in X_test_processed
    if len(feature_names) != X_test_processed.shape[1]:
        print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features ({X_test_processed.shape[1]})")
        feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)

    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Plot ROC curve
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)

    if n_classes == 2:
        # Binary classification
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        if show_plots:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.show()
    else:
        # Multiclass classification
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        y_pred_proba = model.predict_proba(X_test_processed)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        if show_plots:
            plt.figure(figsize=(8, 6))
            colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_classes))
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of class {unique_classes[i]} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
            plt.legend(loc="lower right")
            plt.show()

    return model, X_test_processed, y_test, final_preprocessor

# Example usage:
# Assuming 'data' is your pandas DataFrame and 'target_column' is the name of your target variable
# model, X_test, y_test, preprocessor = prepare_and_train_logistic_regression(data, 'target_column')