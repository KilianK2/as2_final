import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def load_feature_data(file_path):
    """Load the extracted features from CSV"""
    print(f"Loading feature data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} windows with {len(data.columns)} features")
    
    # Print basic statistics
    print(f"Activities in dataset: {data['activity'].unique().tolist()}")
    print(f"Sessions in dataset: {data['session_id'].unique().tolist()}")
    
    activity_counts = data['activity'].value_counts()
    print("\nActivity distribution:")
    for activity, count in activity_counts.items():
        print(f"- {activity}: {count} windows ({count/len(data)*100:.1f}%)")
    
    return data

def session_aware_train_test_split(data, test_size=0.3, random_state=42):
    """
    Split data for training and testing while keeping entire sessions together.
    This ensures that windows from the same session don't appear in both training and testing sets.
    
    Args:
        data: DataFrame containing the feature data
        test_size: Proportion of data to use for testing (default: 0.3)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split feature and target data
    """
    # Use GroupShuffleSplit to keep sessions together
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Identify feature columns (exclude metadata columns)
    metadata_cols = ['activity', 'session_id', 'window_id', 'window_start', 'window_end', 'sample_count']
    feature_cols = [col for col in data.columns if col not in metadata_cols]
    
    # Features (X) and target (y)
    X = data[feature_cols]
    y = data['activity']
    groups = data['session_id']  # Group by session_id
    
    # Get train and test indices
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    # Split the data
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    # Check which sessions are in train and test sets
    train_sessions = data.iloc[train_idx]['session_id'].unique()
    test_sessions = data.iloc[test_idx]['session_id'].unique()
    
    print(f"\nTrain set contains sessions: {train_sessions.tolist()}")
    print(f"Test set contains sessions: {test_sessions.tolist()}")
    
    # Verify that no session appears in both sets
    common_sessions = set(train_sessions) & set(test_sessions)
    if common_sessions:
        print(f"WARNING: {len(common_sessions)} sessions appear in both train and test sets!")
    else:
        print("GOOD: No sessions overlap between train and test sets")
    
    print(f"\nTrain set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Return the split data
    return X_train, X_test, y_train, y_test

def save_train_test_splits(X_train, X_test, y_train, y_test, data, output_dir):
    """
    Save the train and test splits as CSV files with features and labels.
    
    Args:
        X_train, X_test: DataFrames containing training and testing features
        y_train, y_test: Series containing training and testing labels
        data: Original complete DataFrame with metadata
        output_dir: Directory to save the CSV files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metadata columns for both train and test sets
    metadata_cols = ['activity', 'session_id', 'window_id', 'window_start', 'window_end', 'sample_count']
    
    # Get indices from X_train and X_test
    train_indices = X_train.index
    test_indices = X_test.index
    
    # Create complete train and test DataFrames with metadata
    train_data = pd.concat([data.loc[train_indices, metadata_cols], X_train], axis=1)
    test_data = pd.concat([data.loc[test_indices, metadata_cols], X_test], axis=1)
    
    # Save to CSV
    train_file = os.path.join(output_dir, "train_data.csv")
    test_file = os.path.join(output_dir, "test_data.csv")
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"Saved train data ({len(train_data)} samples) to {train_file}")
    print(f"Saved test data ({len(test_data)} samples) to {test_file}")

def load_train_test_data(train_file, test_file):
    """
    Load pre-split train and test data from CSV files.
    
    Args:
        train_file: Path to the training data CSV
        test_file: Path to the testing data CSV
        
    Returns:
        X_train, X_test, y_train, y_test: Split feature and target data
    """
    print(f"Loading pre-split train data from {train_file}...")
    train_data = pd.read_csv(train_file)
    
    print(f"Loading pre-split test data from {test_file}...")
    test_data = pd.read_csv(test_file)
    
    # Identify metadata columns
    metadata_cols = ['activity', 'session_id', 'window_id', 'window_start', 'window_end', 'sample_count']
    feature_cols = [col for col in train_data.columns if col not in metadata_cols]
    
    # Extract features and targets
    X_train = train_data[feature_cols]
    y_train = train_data['activity']
    
    X_test = test_data[feature_cols]
    y_test = test_data['activity']
    
    print(f"Loaded train set: {len(X_train)} samples")
    print(f"Loaded test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def main():
    # Use absolute paths for robust file location
    # Get the absolute path to the project root directory (one level up from the current script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct absolute path to the feature file
    feature_file = os.path.join(project_root, "04_feature extraction", "data_by_features_per_window.csv")
    
    print(f"Project root directory: {project_root}")
    print(f"Looking for feature file at: {feature_file}")
    
    # Check if file exists
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found at: {feature_file}")
    
    data = load_feature_data(feature_file)
    
    # Perform session-aware train-test split
    X_train, X_test, y_train, y_test = session_aware_train_test_split(data, test_size=0.3, random_state=42)
    
    print("\nData successfully split into training and testing sets while keeping sessions together.")
    
    # Print activity distribution in train/test sets
    print("\nActivity distribution in train set:")
    train_activity_counts = y_train.value_counts()
    for activity, count in train_activity_counts.items():
        print(f"- {activity}: {count} windows ({count/len(y_train)*100:.1f}%)")
    
    print("\nActivity distribution in test set:")
    test_activity_counts = y_test.value_counts()
    for activity, count in test_activity_counts.items():
        print(f"- {activity}: {count} windows ({count/len(y_test)*100:.1f}%)")
    
    # Save the train and test splits to CSV files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_train_test_splits(X_train, X_test, y_train, y_test, data, current_dir)
    
    print("\nTrain and test data have been saved as separate CSV files for easy reuse.")

if __name__ == "__main__":
    main()
