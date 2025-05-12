import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
import time

# Add parent directory to path to import splitting_data
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_splitting'))
from splitting_data import load_feature_data, session_aware_train_test_split, load_train_test_data

def train_random_forest_with_cv(X_train, y_train, groups_train=None, cv=5, random_state=42):
    """
    Train a Random Forest classifier with hyperparameter tuning using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        groups_train: Session IDs for group-based cross-validation
        cv: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility
    
    Returns:
        best_model: Trained model with the best hyperparameters
        best_params: Best hyperparameters found
    """
    print("Starting Random Forest training with hyperparameter tuning...")
    start_time = time.time()
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=random_state)
    
    # Set up cross-validation strategy
    if groups_train is not None:
        # Use GroupKFold if session IDs are provided
        print("Using GroupKFold cross-validation to keep sessions together...")
        cv_strategy = GroupKFold(n_splits=cv)
        cv_generator = cv_strategy.split(X_train, y_train, groups=groups_train)
    else:
        # Otherwise use regular K-fold cross-validation
        cv_generator = cv
    
    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_generator,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores for parallel processing
        verbose=1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Print results
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nBest Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    
    # Display training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return best_model, best_params

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names (default: None)
    """
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique class names for plot labels
    if class_names is None:
        unique_classes = sorted(np.unique(y_test))
    else:
        # Ensure class_names is a list, not an array (to avoid ambiguity)
        unique_classes = sorted(class_names) if isinstance(class_names, list) else sorted(list(class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes,
                yticklabels=unique_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot (using model-specific results folder)
    results_folder = os.path.join('results', 'random_forest')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
    plt.close()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Display top features
        print("\nTop 20 Most Important Features:")
        print(feature_importances.head(20))
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'feature_importances.png'))
        plt.close()
        
        # Save feature importances to CSV
        feature_importances.to_csv(os.path.join(results_folder, 'feature_importances.csv'), index=False)

def save_model(model, output_path):
    """Save the trained model to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel saved to {output_path}")

def main():
    # Use absolute paths for robust file location
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct absolute paths to the train and test files
    train_file = os.path.join(project_root, "05_splitting", "train_data.csv")
    test_file = os.path.join(project_root, "05_splitting", "test_data.csv")
    
    print(f"Project root: {project_root}")
    print(f"Looking for train file at: {train_file}")
    print(f"Looking for test file at: {test_file}")
    
    # First check if the pre-split files exist
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Using pre-split train and test datasets")
        X_train, X_test, y_train, y_test = load_train_test_data(train_file, test_file)
    else:
        # Fall back to on-the-fly splitting if files don't exist
        print("Pre-split datasets not found. Performing splitting on the fly...")
        
        # Construct absolute path to the feature file
        feature_file = os.path.join(project_root, "04_feature extraction", "data_by_features_per_window.csv")
        print(f"Looking for feature file at: {feature_file}")
        
        # Check if file exists
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found at: {feature_file}")
            
        data = load_feature_data(feature_file)
        
        # Perform session-aware train-test split
        X_train, X_test, y_train, y_test = session_aware_train_test_split(data, test_size=0.3, random_state=42)
    
    # Apply normalization using StandardScaler
    print("Applying StandardScaler normalization to features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names for feature importance visualization
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Calculate statistics on the normalized data (fix the numpy std issue by specifying axis)
    mean_first_features = np.mean(X_train_scaled.iloc[:, :5].values)
    std_first_features = np.std(X_train_scaled.iloc[:, :5].values, axis=0).mean()
    
    print(f"Features normalized. Mean of first 5 features: {mean_first_features:.5f}")
    print(f"Standard deviation of first 5 features: {std_first_features:.5f}")
    
    # Get session IDs for group-based cross-validation
    # Extract session IDs from the training data
    groups_train = X_train.index.get_level_values('session_id') if 'session_id' in X_train.index.names else None
    
    # If session_id is not in the index, try to get it from the train data dataframe
    if groups_train is None and hasattr(X_train, 'session_id'):
        groups_train = X_train['session_id'].values
    
    # Train model with hyperparameter tuning - now using normalized data
    best_model, best_params = train_random_forest_with_cv(
        X_train_scaled, y_train, 
        groups_train=groups_train,  # Use session IDs for GroupKFold
        cv=5, 
        random_state=42
    )
    
    # Evaluate model using normalized test data
    class_names = list(np.unique(y_train))
    evaluate_model(best_model, X_test_scaled, y_test, class_names)
    
    # Create model-specific results directory
    results_folder = os.path.join('results', 'random_forest')
    os.makedirs(results_folder, exist_ok=True)
    
    # Save model
    model_output_path = os.path.join(results_folder, 'model.joblib')
    save_model(best_model, model_output_path)
    
    # Save scaler for future predictions
    scaler_output_path = os.path.join(results_folder, 'standard_scaler.joblib')
    joblib.dump(scaler, scaler_output_path)
    print(f"Scaler saved to {scaler_output_path}")
    
    # Save hyperparameters
    with open(os.path.join(results_folder, 'hyperparameters.txt'), 'w') as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    print("\nRandom Forest training and evaluation completed successfully!")
    print(f"All results saved to {os.path.abspath(results_folder)}")

if __name__ == "__main__":
    main()
