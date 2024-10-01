import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_graphviz
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pydotplus
import io
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QScrollArea, 
                             QWidget, QTextEdit, QPushButton, QTabWidget)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QScrollArea, 
                             QWidget, QTextEdit, QPushButton, QTabWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

def show_decision_tree(X_train, y_train, feature_names, target_labels, tree_type, params, parent=None):
    if params is None:
        print("Invalid parameters provided")
        return None

    print("show_decision_tree function called")
    tree_widget = QWidget(parent)
    main_layout = QVBoxLayout(tree_widget)

    if params.get("use_grid_search"):
        # Use custom grid search parameters
        param_grid = params.get("grid_search_params", {})
        
        if not param_grid:
            error_message = QLabel("No valid grid search parameters provided.")
            main_layout.addWidget(error_message)
            return tree_widget
        
        if tree_type == "classification":
            base_model = DecisionTreeClassifier(random_state=42)
        else:
            base_model = DecisionTreeRegressor(random_state=42)
        
        grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        clf = grid_search.best_estimator_
        
        # Display best parameters
        best_params_text = QTextEdit()
        best_params_text.setPlainText(f"Best parameters: {best_params}")
        main_layout.addWidget(best_params_text)
    else:
        if tree_type == "classification":
            clf = DecisionTreeClassifier(random_state=42, **params)
        else:
            clf = DecisionTreeRegressor(random_state=42, **params)
        clf.fit(X_train, y_train)

    title = "Decision Tree Classifier" if tree_type == "classification" else "Decision Tree Regressor"

    # Convert y_train to DataFrame if it's a Series
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()

    # Create a figure for each target variable
    for i, target_col in enumerate(y_train.columns):
        y = y_train[target_col]
        if not params["use_grid_search"]:
            clf.fit(X_train, y)
        
        # Determine tree depth and adjust figure size
        n_nodes = clf.tree_.node_count
        depth = clf.tree_.max_depth
        fig_width = np.max([20, depth * 4])
        fig_height = np.max([10, n_nodes * 0.5])
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # Ensure feature_names match the number of features
        if len(feature_names) != X_train.shape[1]:
            print(f"Warning: Number of features ({X_train.shape[1]}) doesn't match number of feature names ({len(feature_names)})")
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Plot the tree
        try:
            plot_tree(clf, feature_names=feature_names, class_names=target_labels,
                    rounded=True, filled=True, fontsize=8,
                    ax=ax, proportion=True, precision=2, impurity=False,
                    node_ids=True, max_depth=5)
        except Exception as e:
            print(f"Error plotting tree: {e}")
            continue
        
        plt.title(f"{title} for {target_col}", fontsize=16, fontweight='bold')
        plt.tight_layout(pad=1.0)
        
        # Add the matplotlib plot
        canvas = FigureCanvas(fig)
        main_layout.addWidget(canvas)
        
        toolbar = NavigationToolbar(canvas, tree_widget)
        main_layout.addWidget(toolbar)
        
        # Add the tree visualization
        tree_png = visualize_tree(clf, feature_names)
        pixmap = QPixmap()
        pixmap.loadFromData(tree_png)
        tree_label = QLabel()
        tree_label.setPixmap(pixmap.scaledToWidth(1200, Qt.SmoothTransformation))
        
        # Add scroll area for the tree visualization
        scroll_area = QScrollArea()
        scroll_area.setWidget(tree_label)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Start with a slightly zoomed out view
        ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)

    return tree_widget

def visualize_tree(model, feature_names):
    dot_data = io.StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=feature_names,
                    filled=True, rounded=True, special_characters=True,
                    max_depth=5,  # Limit depth to reduce complexity
                    proportion=True, precision=2, impurity=False,
                    rotate=True)  # Rotate tree to horizontal layout
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.set_size('"12,8!"')  # Set size of the image (! makes it minimum size)
    graph.set_rankdir('LR')  # Set direction from left to right
    return graph.create_png()

def show_decision_trees(X_train, y_train, feature_names, target_labels, tree_type, params, parent=None):
    if params is None:
        print("Invalid parameters provided")
        return None

    print("show_decision_trees function called")
    tree_widget = QWidget(parent)
    main_layout = QVBoxLayout(tree_widget)

    if params.get("use_grid_search"):
        param_grid = params.get("grid_search_params", {})
        
        if not param_grid:
            error_message = QLabel("No valid grid search parameters provided.")
            main_layout.addWidget(error_message)
            return tree_widget
        
        if tree_type == "classification":
            base_model = DecisionTreeClassifier(random_state=42)
        else:
            base_model = DecisionTreeRegressor(random_state=42)
        
        grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_clf = grid_search.best_estimator_
        
        # Display best parameters
        best_params_text = QTextEdit()
        best_params_text.setPlainText(f"Best parameters: {best_params}")
        main_layout.addWidget(best_params_text)

        # Visualize best tree
        best_tree_widget = create_tree_visualization(best_clf, X_train, y_train, feature_names, target_labels, f"Best {tree_type.capitalize()} Tree")
        main_layout.addWidget(best_tree_widget)

        # Add button to show all trees
        show_all_button = QPushButton("Show All Trees")
        show_all_button.clicked.connect(lambda: show_all_trees(grid_search, X_train, y_train, feature_names, target_labels, tree_type))
        main_layout.addWidget(show_all_button)

    else:
        if tree_type == "classification":
            clf = DecisionTreeClassifier(random_state=42, **params)
        else:
            clf = DecisionTreeRegressor(random_state=42, **params)
        clf.fit(X_train, y_train)

        tree_widget = create_tree_visualization(clf, X_train, y_train, feature_names, target_labels, f"{tree_type.capitalize()} Tree")
        main_layout.addWidget(tree_widget)

    # Always add the button to show multiple trees
    show_multiple_button = QPushButton("Show Multiple Decision Trees")
    show_multiple_button.clicked.connect(lambda: show_multiple_trees(X_train, y_train, feature_names, target_labels, tree_type, params))
    main_layout.addWidget(show_multiple_button)

    return tree_widget

def create_tree_visualization(clf, X_train, y_train, feature_names, target_labels, title):
    tree_widget = QWidget()
    layout = QVBoxLayout(tree_widget)

    # Determine tree depth and adjust figure size
    n_nodes = clf.tree_.node_count
    depth = clf.tree_.max_depth
    fig_width = np.max([20, depth * 4])
    fig_height = np.max([10, n_nodes * 0.5])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Ensure feature_names match the number of features
    if len(feature_names) != X_train.shape[1]:
        print(f"Warning: Number of features ({X_train.shape[1]}) doesn't match number of feature names ({len(feature_names)})")
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Plot the tree
    try:
        plot_tree(clf, feature_names=feature_names, class_names=target_labels,
                  rounded=True, filled=True, fontsize=8,
                  ax=ax, proportion=True, precision=2, impurity=False,
                  node_ids=True, max_depth=5)
    except Exception as e:
        print(f"Error plotting tree: {e}")
        error_label = QLabel(f"Error plotting tree: {e}")
        layout.addWidget(error_label)
        return tree_widget

    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.0)

    # Add the matplotlib plot
    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)

    toolbar = NavigationToolbar(canvas, tree_widget)
    layout.addWidget(toolbar)

    # Add the tree visualization
    tree_png = visualize_tree(clf, feature_names)
    pixmap = QPixmap()
    pixmap.loadFromData(tree_png)
    tree_label = QLabel()
    tree_label.setPixmap(pixmap.scaledToWidth(1200, Qt.SmoothTransformation))

    # Add scroll area for the tree visualization
    scroll_area = QScrollArea()
    scroll_area.setWidget(tree_label)
    scroll_area.setWidgetResizable(True)
    layout.addWidget(scroll_area)

    # Start with a slightly zoomed out view
    ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)

    return tree_widget

def show_all_trees(grid_search, X_train, y_train, feature_names, target_labels, tree_type):
    all_trees_dialog = QDialog()
    all_trees_layout = QVBoxLayout(all_trees_dialog)

    tab_widget = QTabWidget()

    for params, estimator in zip(grid_search.cv_results_['params'], grid_search.cv_results_['estimator']):
        tree_widget = create_tree_visualization(estimator, X_train, y_train, feature_names, target_labels, f"{tree_type.capitalize()} Tree")
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        tab_widget.addTab(tree_widget, param_str)

    all_trees_layout.addWidget(tab_widget)
    
    close_button = QPushButton("Close")
    close_button.clicked.connect(all_trees_dialog.close)
    all_trees_layout.addWidget(close_button)

    all_trees_dialog.setLayout(all_trees_layout)
    all_trees_dialog.resize(1000, 800)
    all_trees_dialog.exec_()