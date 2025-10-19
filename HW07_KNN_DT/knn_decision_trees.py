"""
HW06: Exploring KNN and Decision Trees - Starter Code
This script provides the foundation for Tasks 1-5.

You will implement Tasks 6-7 as part of the assignment.

Design: 

Intention -- Separates data processing/model training from visualization
to enable easy unit testing and grading automation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import graphviz
import os
import seaborn as sns

from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score) 


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Task 1: Load the wine dataset and prepare it for modeling.
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        dict: {
            'X_train': Training features (scaled),
            'X_test': Test features (scaled),
            'y_train': Training labels,
            'y_test': Test labels,
            'feature_names': List of feature names,
            'target_names': List of class names,
            'scaler': Fitted StandardScaler object
        }
    """
    # Load the wine dataset
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_names': target_names,
        'scaler': scaler
    }


def load_and_prepare_iris_data(test_size=0.2, random_state=42):
    """
    Load the Iris dataset and prepare it for modeling.
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        dict: {
            'X_train': Training features (scaled),
            'X_test': Test features (scaled),
            'y_train': Training labels,
            'y_test': Test labels,
            'feature_names': List of feature names,
            'target_names': List of class names,
            'scaler': Fitted StandardScaler object
        }
    """
    # Load the iris dataset
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_names': target_names,
        'scaler': scaler
    }


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_knn_single(X_train, X_test, y_train, y_test, k=5):
    """
    Train a single KNN model and return predictions and accuracy.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        k: Number of neighbors
    
    Returns:
        dict: {
            'model': Trained KNeighborsClassifier,
            'y_pred': Predictions on test set,
            'accuracy': Accuracy score,
            'k': Value of k used
        }
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'k': k
    }


def train_knn_range(X_train, X_test, y_train, y_test, k_range=range(1, 11)):
    """
    Task 2: Train KNN models with varying k values.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        k_range: Range of k values to test
    
    Returns:
        dict: {
            'k_values': List of k values tested,
            'accuracies': List of corresponding accuracies,
            'models': List of trained models,
            'predictions': List of predictions for each k
        }
    """
    k_values = list(k_range)
    accuracies = []
    models = []
    predictions = []
    
    for k in k_values:
        result = train_knn_single(X_train, X_test, y_train, y_test, k)
        accuracies.append(result['accuracy'])
        models.append(result['model'])
        predictions.append(result['y_pred'])
    
    return {
        'k_values': k_values,
        'accuracies': accuracies,
        'models': models,
        'predictions': predictions
    }


def train_decision_tree_single(X_train, X_test, y_train, y_test, max_depth=5):
    """
    Train a single Decision Tree model and return predictions and accuracy.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        max_depth: Maximum depth of the tree
    
    Returns:
        dict: {
            'model': Trained DecisionTreeClassifier,
            'y_pred': Predictions on test set,
            'accuracy': Accuracy score,
            'depth': Maximum depth used
        }
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'depth': max_depth
    }


def train_decision_tree_range(X_train, X_test, y_train, y_test, depth_range=range(1, 11)):
    """
    Task 4: Train Decision Tree models with varying depths.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        depth_range: Range of depth values to test
    
    Returns:
        dict: {
            'depths': List of depths tested,
            'accuracies': List of corresponding accuracies,
            'models': List of trained models,
            'predictions': List of predictions for each depth,
            'best_model': Model with highest accuracy,
            'best_depth': Depth of best model
        }
    """
    depths = list(depth_range)
    accuracies = []
    models = []
    predictions = []
    best_accuracy = 0
    best_model = None
    best_depth = None
    
    for depth in depths:
        result = train_decision_tree_single(X_train, X_test, y_train, y_test, depth)
        accuracies.append(result['accuracy'])
        models.append(result['model'])
        predictions.append(result['y_pred'])
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model = result['model']
            best_depth = depth
    
    return {
        'depths': depths,
        'accuracies': accuracies,
        'models': models,
        'predictions': predictions,
        'best_model': best_model,
        'best_depth': best_depth
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_knn_decision_boundary_data(X_train, y_train, model, resolution=0.02):
    """
    Generate data for KNN decision boundary visualization.
    Uses only the first two features.
    
    Args:
        X_train: Training features (will use only first 2)
        y_train: Training labels
        model: Trained KNN model (should be trained on 2 features)
        resolution: Step size for mesh grid
    
    Returns:
        dict: {
            'xx': X coordinates of mesh grid,
            'yy': Y coordinates of mesh grid,
            'Z': Predicted classes for mesh grid,
            'X_vis': Training data (first 2 features),
            'y_vis': Training labels
        }
    """
    X_vis = X_train[:, :2]
    
    # Create mesh grid
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    return {
        'xx': xx,
        'yy': yy,
        'Z': Z,
        'X_vis': X_vis,
        'y_vis': y_train
    }


def plot_knn_decision_boundaries(boundary_data, feature_names, k, show=True, save_path=None):
    """
    Task 3: Visualize KNN decision boundaries using the first two features.
    
    Args:
        boundary_data: Dictionary from generate_knn_decision_boundary_data()
        feature_names: Names of the features (will use first 2)
        k: Value of k for the KNN model
        show: Whether to display the plot
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    xx = boundary_data['xx']
    yy = boundary_data['yy']
    Z = boundary_data['Z']
    X_vis = boundary_data['X_vis']
    y_vis = boundary_data['y_vis']
    
    # Color schemes
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#00FF00', '#0000FF']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    
    # Plot training points
    for i, color in zip(np.unique(y_vis), cmap_bold):
        idx = np.where(y_vis == i)
        ax.scatter(
            X_vis[idx, 0], X_vis[idx, 1],
            c=color, label=f"Class {i}",
            edgecolor='black', s=50
        )
    
    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title(f"KNN Decision Boundaries (k={k})", fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def export_tree_to_image(tree, feature_names, class_names, output_path="decision_tree"):
    """
    Export decision tree to an image file.
    
    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: Names of features
        class_names: Names of classes
        output_path: Path for output file (without extension)
    
    Returns:
        str: Path to the generated PNG file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export to DOT format
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    # Render as PNG
    graph = graphviz.Source(dot_data, format='png')
    graph.render(output_path, format='png', cleanup=True)
    
    return f"{output_path}.png"


def visualize_tree(tree, feature_names, class_names, show=True, save_path="decision_tree"):
    """
    Task 5: Visualize the Decision Tree structure.
    
    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: Names of features
        class_names: Names of classes
        show: Whether to display the plot
        save_path: Path to save the tree image
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Export tree to image
    img_path = export_tree_to_image(tree, feature_names, class_names, save_path)
    
    # Load and display
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Decision Tree Structure", fontsize=16, pad=20)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_model_performance(k_values, knn_accuracies, depths, tree_accuracies, 
                          show=True, save_path=None):
    """
    Compare KNN and Decision Tree performance across different parameters.
    
    Args:
        k_values: List of k values tested for KNN
        knn_accuracies: Corresponding accuracies for KNN
        depths: List of depths tested for Decision Trees
        tree_accuracies: Corresponding accuracies for Decision Trees
        show: Whether to display the plot
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(k_values, knn_accuracies, marker='o', linewidth=2, 
            markersize=8, label='KNN', color='#1f77b4')
    ax.plot(depths, tree_accuracies, marker='s', linewidth=2, 
            markersize=8, label='Decision Tree', color='#ff7f0e')
    
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xlabel('Parameter Value (k for KNN, Depth for Tree)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(range(1, max(max(k_values), max(depths)) + 1))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# REPORTING FUNCTIONS (for display)
# =============================================================================

def print_data_summary(data_dict):
    """Print a summary of the loaded dataset."""
    print(f"Dataset Summary:")
    print(f"  Training samples: {len(data_dict['y_train'])}")
    print(f"  Test samples: {len(data_dict['y_test'])}")
    print(f"  Features: {len(data_dict['feature_names'])}")
    print(f"  Classes: {len(data_dict['target_names'])}")


def print_knn_results(knn_results):
    """Print KNN training results."""
    print("\nKNN Training Results:")
    for k, acc in zip(knn_results['k_values'], knn_results['accuracies']):
        print(f"  k={k:2d}: Accuracy = {acc:.4f}")


def print_tree_results(tree_results):
    """Print Decision Tree training results."""
    print("\nDecision Tree Training Results:")
    for depth, acc in zip(tree_results['depths'], tree_results['accuracies']):
        print(f"  Depth={depth:2d}: Accuracy = {acc:.4f}")
    print(f"\nBest model: Depth={tree_results['best_depth']}, "
          f"Accuracy={max(tree_results['accuracies']):.4f}")


# =============================================================================
# TASK 6: PERFORMANCE METRICS FUNCTIONS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    """Visualize confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred, model_name, class_names=None):
    """Calculate and display comprehensive evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    plot_confusion_matrix(y_true, y_pred, model_name, class_names)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def run_complete_analysis(data, dataset_name):
    """
    Run complete analysis (Tasks 1-6) on a given dataset.
    
    Args:
        data: Dataset dictionary from load_and_prepare_data functions
        dataset_name: Name of the dataset for display purposes
    
    Returns:
        dict: Results containing best models and metrics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name} Dataset")
    print(f"{'='*60}")
    
    # Print dataset summary
    print_data_summary(data)
    
    # Train KNN models
    print(f"\n[KNN] Training KNN models on {dataset_name}...")
    knn_results = train_knn_range(
        data['X_train'], data['X_test'], 
        data['y_train'], data['y_test']
    )
    print_knn_results(knn_results)
    
    # Train Decision Tree models
    print(f"\n[Decision Tree] Training Decision Tree models on {dataset_name}...")
    tree_results = train_decision_tree_range(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test']
    )
    print_tree_results(tree_results)
    
    # Evaluate models
    print(f"\n[Evaluation] Evaluating models on {dataset_name}...")
    knn_metrics = []
    for i, (k, predictions) in enumerate(zip(knn_results['k_values'], knn_results['predictions'])):
        metrics = evaluate_model(
            data['y_test'], 
            predictions, 
            f"KNN (k={k})", 
            data['target_names']
        )
        knn_metrics.append({'k': k, 'metrics': metrics})
    
    tree_metrics = []
    for i, (depth, predictions) in enumerate(zip(tree_results['depths'], tree_results['predictions'])):
        metrics = evaluate_model(
            data['y_test'], 
            predictions, 
            f"Decision Tree (depth={depth})", 
            data['target_names']
        )
        tree_metrics.append({'depth': depth, 'metrics': metrics})
    
    # Find best models
    best_knn = max(knn_metrics, key=lambda x: x['metrics']['f1_score'])
    best_tree = max(tree_metrics, key=lambda x: x['metrics']['f1_score'])
    
    print(f"\n🏆 Best KNN Model: k={best_knn['k']} (F1={best_knn['metrics']['f1_score']:.4f})")
    print(f"🏆 Best Decision Tree Model: depth={best_tree['depth']} (F1={best_tree['metrics']['f1_score']:.4f})")
    
    return {
        'dataset_name': dataset_name,
        'knn_results': knn_results,
        'tree_results': tree_results,
        'knn_metrics': knn_metrics,
        'tree_metrics': tree_metrics,
        'best_knn': best_knn,
        'best_tree': best_tree
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function that runs all tasks.
    """
    print("="*60)
    print("HW06: Exploring KNN and Decision Trees")
    print("="*60)
    
    # Task 1: Load and prepare data
    print("\n[Task 1] Loading and preparing data...")
    data = load_and_prepare_data()
    print_data_summary(data)
    
    # Task 2: Train KNN models
    print("\n[Task 2] Training KNN models...")
    knn_results = train_knn_range(
        data['X_train'], data['X_test'], 
        data['y_train'], data['y_test']
    )
    print_knn_results(knn_results)
    
    # Task 3: Visualize KNN decision boundaries
    print("\n[Task 3] Creating KNN decision boundary visualization (k=5)...")
    knn_model_2d = KNeighborsClassifier(n_neighbors=5)
    knn_model_2d.fit(data['X_train'][:, :2], data['y_train'])
    boundary_data = generate_knn_decision_boundary_data(
        data['X_train'], data['y_train'], knn_model_2d
    )
    plot_knn_decision_boundaries(
        boundary_data, data['feature_names'][:2], k=5
    )
    
    # Task 4: Train Decision Tree models
    print("\n[Task 4] Training Decision Tree models...")
    tree_results = train_decision_tree_range(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test']
    )
    print_tree_results(tree_results)
    
    # Task 5: Visualize Decision Tree structure
    print("\n[Task 5] Visualizing Decision Tree structure...")
    visualize_tree(
        tree_results['best_model'],
        data['feature_names'],
        [str(name) for name in data['target_names']]
    )
    
    # Compare model performance
    print("\n[Comparison] Plotting model performance...")
    plot_model_performance(
        knn_results['k_values'], knn_results['accuracies'],
        tree_results['depths'], tree_results['accuracies']
    )
    
    print("\n" + "="*60)
    print("✅ Tasks 1-5 complete!")
    print("="*60)
    
    # ========================================================================
    # TODO: TASK 6 - Implement Performance Metrics Analysis
    # ========================================================================
    # Your task: Implement confusion matrix analysis and calculate metrics
    # 
    # Requirements:
    # 1. Create a function to plot confusion matrices (use seaborn heatmap)
    # 2. Create a function to evaluate models with multiple metrics:
    #    - Accuracy
    #    - Precision
    #    - Recall
    #    - F1 Score
    #    - Specificity (for binary classification)
    # 3. Evaluate both KNN and Decision Tree models
    # 4. Compare performance across different k values and tree depths
    # 5. Analyze and discuss misclassifications
    #
    # Hints:
    # - Import: from sklearn.metrics import confusion_matrix, precision_score, 
    #           recall_score, f1_score
    # - Import: import seaborn as sns
    # - Use average='weighted' for multi-class precision, recall, and F1
    # - Refer to the lab description for code templates
    # - You can access model predictions from knn_results['predictions'] 
    #   and tree_results['predictions']
    # ========================================================================
    
    print("\n📝 TODO: Implement Task 6 - Performance Metrics Analysis")
    print("   See code comments above for requirements")
    print("   Available data:")
    print(f"     - data['y_test']: true labels")
    print(f"     - knn_results['predictions']: KNN predictions for each k")
    print(f"     - tree_results['predictions']: Tree predictions for each depth")
    
    # Task 6: Performance Metrics Analysis Implementation
    
    # Evaluate KNN models
    print("\n[Task 6] Evaluating KNN models...")
    knn_metrics = []
    for i, (k, predictions) in enumerate(zip(knn_results['k_values'], knn_results['predictions'])):
        print(f"\n--- KNN with k={k} ---")
        metrics = evaluate_model(
            data['y_test'], 
            predictions, 
            f"KNN (k={k})", 
            data['target_names']
        )
        knn_metrics.append({'k': k, 'metrics': metrics})
    
    # Evaluate Decision Tree models
    print("\n[Task 6] Evaluating Decision Tree models...")
    tree_metrics = []
    for i, (depth, predictions) in enumerate(zip(tree_results['depths'], tree_results['predictions'])):
        print(f"\n--- Decision Tree with depth={depth} ---")
        metrics = evaluate_model(
            data['y_test'], 
            predictions, 
            f"Decision Tree (depth={depth})", 
            data['target_names']
        )
        tree_metrics.append({'depth': depth, 'metrics': metrics})
    
    # Find best models based on F1 score
    best_knn = max(knn_metrics, key=lambda x: x['metrics']['f1_score'])
    best_tree = max(tree_metrics, key=lambda x: x['metrics']['f1_score'])
    
    print(f"\n🏆 Best KNN Model: k={best_knn['k']} (F1={best_knn['metrics']['f1_score']:.4f})")
    print(f"🏆 Best Decision Tree Model: depth={best_tree['depth']} (F1={best_tree['metrics']['f1_score']:.4f})")
    
    # Performance comparison summary
    print(f"\n📊 Performance Summary:")
    print(f"  Best KNN Accuracy: {best_knn['metrics']['accuracy']:.4f}")
    print(f"  Best Tree Accuracy: {best_tree['metrics']['accuracy']:.4f}")
    print(f"  Best KNN F1 Score: {best_knn['metrics']['f1_score']:.4f}")
    print(f"  Best Tree F1 Score: {best_tree['metrics']['f1_score']:.4f}")
    
    # Misclassification analysis
    print(f"\n🔍 Misclassification Analysis:")
    print(f"  KNN (k={best_knn['k']}) misclassified {sum(data['y_test'] != knn_results['predictions'][knn_results['k_values'].index(best_knn['k'])])} out of {len(data['y_test'])} samples")
    print(f"  Tree (depth={best_tree['depth']}) misclassified {sum(data['y_test'] != tree_results['predictions'][tree_results['depths'].index(best_tree['depth'])])} out of {len(data['y_test'])} samples")
  

    
    # ========================================================================
    # TASK 7 - Dataset Exploration Implementation
    # ========================================================================
    
    print("\n[Task 7] Exploring Iris Dataset...")
    
    # Load Iris dataset
    iris_data = load_and_prepare_iris_data()
    
    # Run complete analysis on Iris dataset
    iris_results = run_complete_analysis(iris_data, "Iris")
    
    # Compare Wine vs Iris results
    print(f"\n{'='*60}")
    print("DATASET COMPARISON: Wine vs Iris")
    print(f"{'='*60}")
    
    print(f"\n📊 Dataset Characteristics:")
    print(f"  Wine Dataset:")
    print(f"    - Samples: {len(data['y_train']) + len(data['y_test'])} total")
    print(f"    - Features: {len(data['feature_names'])}")
    print(f"    - Classes: {len(data['target_names'])}")
    print(f"    - Best KNN: k={best_knn['k']} (F1={best_knn['metrics']['f1_score']:.4f})")
    print(f"    - Best Tree: depth={best_tree['depth']} (F1={best_tree['metrics']['f1_score']:.4f})")
    
    print(f"\n  Iris Dataset:")
    print(f"    - Samples: {len(iris_data['y_train']) + len(iris_data['y_test'])} total")
    print(f"    - Features: {len(iris_data['feature_names'])}")
    print(f"    - Classes: {len(iris_data['target_names'])}")
    print(f"    - Best KNN: k={iris_results['best_knn']['k']} (F1={iris_results['best_knn']['metrics']['f1_score']:.4f})")
    print(f"    - Best Tree: depth={iris_results['best_tree']['depth']} (F1={iris_results['best_tree']['metrics']['f1_score']:.4f})")
    
    print(f"\n🔍 Performance Analysis:")
    print(f"  Wine Dataset Performance:")
    print(f"    - KNN Accuracy: {best_knn['metrics']['accuracy']:.4f}")
    print(f"    - Tree Accuracy: {best_tree['metrics']['accuracy']:.4f}")
    print(f"    - Winner: {'KNN' if best_knn['metrics']['f1_score'] > best_tree['metrics']['f1_score'] else 'Decision Tree'}")
    
    print(f"\n  Iris Dataset Performance:")
    print(f"    - KNN Accuracy: {iris_results['best_knn']['metrics']['accuracy']:.4f}")
    print(f"    - Tree Accuracy: {iris_results['best_tree']['metrics']['accuracy']:.4f}")
    print(f"    - Winner: {'KNN' if iris_results['best_knn']['metrics']['f1_score'] > iris_results['best_tree']['metrics']['f1_score'] else 'Decision Tree'}")
    
    print(f"\n💡 Dataset Characteristics Impact:")
    print(f"  Wine (13 features, 3 classes, 178 samples):")
    print(f"    - Higher dimensional, more complex feature space")
    print(f"    - KNN performs exceptionally well (perfect on test set)")
    print(f"    - Decision trees struggle more with complex boundaries")
    
    print(f"\n  Iris (4 features, 3 classes, 150 samples):")
    print(f"    - Lower dimensional, simpler feature space")
    print(f"    - Both models perform well due to clear class separation")
    print(f"    - Decision trees can capture patterns more easily with fewer features")
    
    print(f"\n🎯 Key Insights:")
    print(f"  1. Dataset dimensionality affects model performance differently")
    print(f"  2. KNN excels with well-separated classes regardless of dimensionality")
    print(f"  3. Decision trees perform better on simpler, lower-dimensional datasets")
    print(f"  4. Feature scaling is crucial for KNN performance")
    print(f"  5. Both datasets show that proper hyperparameter tuning is essential")
    
    
    print("\n" + "="*60)
    print("TODO tasks completed? Nice work!")
    print("Don't forget to:")
    print("  - Write your analysis report")
    print("  - Create your demo video")
    print("  - Post highlights to Piazza")
    print("="*60)


if __name__ == "__main__":
    main()
