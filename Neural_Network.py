import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score, accuracy_score
from scipy.stats import loguniform
import tensorflow as tf
from scikeras.wrappers import KerasRegressor, KerasClassifier
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV


# Import required modules using the tf alias
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam
SGD = tf.keras.optimizers.SGD
RMSprop = tf.keras.optimizers.RMSprop
EarlyStopping = tf.keras.callbacks.EarlyStopping
Sequential = tf.keras.Sequential

logging.basicConfig(level=logging.INFO)

def create_model(learning_rate=0.001, optimizer='Adam', layers=[(32, 'relu')], input_shape=None, output_shape=None, problem_type='regression'):
    
    inputs = tf.keras.Input(shape=(input_shape,))
    x = inputs
    for neurons, activation in layers:
        x = Dense(neurons, activation=activation)(x)
    
    if problem_type == 'regression':
        outputs = Dense(output_shape)(x)
        loss = 'mean_squared_error'
        metrics = ['mean_absolute_error']
    else:  # classification
        if output_shape == 1:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = Dense(output_shape, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def prepare_and_train_neural_network(data, params):
    logging.info("Starting neural network preparation and training")
    
    # Initialize result dictionary
    result = {}
    
    scaler_y = None  # Initialize scaler_y as None
    # Extract parameters
    test_size = params.get('test_size', 0.2)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 32)
    train_columns = params.get('train_columns', [])
    test_columns = params.get('test_columns', [])
    use_grid_search = params.get('use_grid_search', False)
    grid_search_params = params.get('grid_search_params', {})

    # Validate input data
    if not train_columns:
        raise ValueError("No training columns selected.")
    if not test_columns:
        raise ValueError("No test columns selected.")

    # Separate features and target
    X = data[train_columns]
    y = data[test_columns]

    # Determine if it's a regression or classification problem
    is_regression = all(y[col].dtype in ['float64', 'float32', 'int64', 'int32'] for col in y.columns)
    for col in y.columns:
        if y[col].dtype == 'object' or y[col].nunique() < 10:
            is_regression = False
            break

    problem_type = 'regression' if is_regression else 'classification'
    logging.info(f"Detected problem type: {problem_type}")

    # Handle NaN values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create preprocessor
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Scale the target variables
    if is_regression:
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        if y.shape[1] == 1:  # Binary classification
            le = LabelEncoder()
            y_train_scaled = le.fit_transform(y_train.values.ravel())
            y_test_scaled = le.transform(y_test.values.ravel())
        else:  # Multi-class classification
            ohe = OneHotEncoder(sparse_output=False)
            y_train_scaled = ohe.fit_transform(y_train)
            y_test_scaled = ohe.transform(y_test)

    # Ensure y is 2D
    if y_train_scaled.ndim == 1:
        y_train_scaled = y_train_scaled.reshape(-1, 1)
    if y_test_scaled.ndim == 1:
        y_test_scaled = y_test_scaled.reshape(-1, 1)

    input_shape = X_train_processed.shape[1]
    output_shape = y_train_scaled.shape[1]

    # Define the model creation function
    def create_model(optimizer='Adam', layers=[(64, 'relu'), (32, 'relu')]):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_shape,)))
        for neurons, activation in layers:
            model.add(tf.keras.layers.Dense(neurons, activation=activation))
        
        if is_regression:
            model.add(tf.keras.layers.Dense(output_shape, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        elif output_shape == 1:  # Binary classification
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # Multi-class classification
            model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    # In prepare_and_train_neural_network function
    # Add these debug prints before the grid search section
    print("Shape of X_train_processed:", X_train_processed.shape)
    print("Shape of y_train_scaled:", y_train_scaled.shape)
    print("Data types:", X_train_processed.dtype, y_train_scaled.dtype)

    if use_grid_search:
        # Define the model wrapper
        model_wrapper = KerasRegressor if is_regression else KerasClassifier
        model = model_wrapper(
            model=create_model,  # Changed from build_fn to model
            optimizer=tf.keras.optimizers.Adam(),
            verbose=0
        )

        # Define the parameter grid
        param_grid = {
            'optimizer__learning_rate': grid_search_params.get('learning_rate', [0.001, 0.01, 0.1]),
            'batch_size': grid_search_params.get('batch_size', [16, 32, 64]),
            'epochs': grid_search_params.get('epochs', [50, 100, 150])
        }

        # Perform randomized search
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                        n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
        
        try:
            random_search_result = random_search.fit(X_train_processed, y_train_scaled)
            best_model = random_search_result.best_estimator_.model
            best_params = random_search_result.best_params_
            
            # Train the best model
            history = best_model.fit(
                X_train_processed, y_train_scaled,
                validation_split=0.2,
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size'],
                verbose=1
            )

            # Get all results
            all_results = random_search_result.cv_results_
            
            # Calculate accuracy for each hyperparameter set
            accuracies = []
            for params, mean_score in zip(all_results['params'], all_results['mean_test_score']):
                accuracies.append({'params': params, 'accuracy': mean_score})
            
            # Store results in the result dictionary
            result['all_results'] = all_results
            result['best_params'] = best_params
            result['accuracies'] = accuracies
            
        except Exception as e:
            print(f"Error during grid search: {e}")
            use_grid_search = False
            best_model = None
            best_params = None
            all_results = None

    if not use_grid_search or best_model is None:
        # Create and train the model without hyperparameter tuning
        optimizer = params.get('optimizer', 'Adam')
        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))
        elif optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=params.get('learning_rate', 0.001))
        elif optimizer == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=params.get('learning_rate', 0.001))
        
        model = create_model(
            optimizer=optimizer,
            layers=params.get('layers', [(64, 'relu'), (32, 'relu')])
        )
        history = model.fit(
            X_train_processed, y_train_scaled,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        best_model = model
        best_params = None
        all_results = None

    # Predict and calculate metrics
    y_pred = best_model.predict(X_test_processed)
    
    metrics = {}
    for i, col in enumerate(test_columns):
        if is_regression:
            mse = mean_squared_error(y_test_scaled[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_scaled[:, i], y_pred[:, i])
            r2 = r2_score(y_test_scaled[:, i], y_pred[:, i])
            metrics[col] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
        else:
            if y.shape[1] == 1:
                f1 = f1_score(y_test_scaled, (y_pred > 0.5).astype(int), average='weighted')
                accuracy = accuracy_score(y_test_scaled, (y_pred > 0.5).astype(int))
            else:
                f1 = f1_score(y_test_scaled.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
                accuracy = accuracy_score(y_test_scaled.argmax(axis=1), y_pred.argmax(axis=1))
            metrics[col] = {'f1_score': f1, 'accuracy': accuracy}

    logging.info("Neural network preparation and training completed")
    return_dict = {
        "model": best_model if use_grid_search else model,
        "history": history,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "scaler_y": scaler_y,
        "is_regression": is_regression,
        "metrics": metrics,
        "best_params": best_params,
        "all_results": all_results,
        "accuracies": accuracies if use_grid_search else None
    }

    return return_dict