import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict
from datetime import datetime

# Add parent directories to path to import feature extraction code
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '04_feature extraction'))
from feature_extraction import extract_features

def load_random_forest_model():
    """Load the pretrained Random Forest model and scaler"""
    model_path = os.path.join('results', 'random_forest', 'model.joblib')
    scaler_path = os.path.join('results', 'random_forest', 'standard_scaler.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def load_imu_data(file_path):
    """Load IMU data from CSV file"""
    print(f"Loading IMU data from {file_path}")
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} samples")
    
    # Check if necessary columns exist
    required_columns = ['GX', 'GY', 'GZ', 'AX', 'AY', 'AZ', 'timestamp']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    return data

def create_windows(data, window_size_seconds=2.0, overlap_percent=50):
    """Split data into windows with specified overlap"""
    print(f"Creating windows with size={window_size_seconds}s and overlap={overlap_percent}%")
    
    # Sort data by timestamp
    data = data.sort_values('timestamp')
    
    # Calculate window parameters
    timestamps = data['timestamp'].values
    if len(timestamps) == 0:
        return []
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    total_duration = end_time - start_time
    
    print(f"Data spans {total_duration:.2f} seconds from {start_time} to {end_time}")
    
    # Convert window size to seconds
    window_size = window_size_seconds  # in seconds
    step_size = window_size * (1 - overlap_percent / 100)  # in seconds
    
    # Create windows
    window_starts = np.arange(start_time, end_time - window_size + 0.1, step_size)
    print(f"Creating {len(window_starts)} windows")
    
    all_windows = []
    # Process each window
    for i, window_start in enumerate(window_starts):
        window_end = window_start + window_size
        
        # Get data in this time window
        window_data = data[(data['timestamp'] >= window_start) & 
                        (data['timestamp'] < window_end)]
        
        # Only process windows with sufficient data
        if len(window_data) < 10:  # Minimum samples for a useful window
            continue
        
        # Extract features for this window
        features = extract_features(window_data)
        
        # Add metadata
        features['window_id'] = i
        features['window_start'] = window_start
        features['window_end'] = window_end
        features['sample_count'] = len(window_data)
        
        all_windows.append(features)
    
    # Combine all windows into a single DataFrame
    if all_windows:
        windows_df = pd.DataFrame(all_windows)
        print(f"Created {len(windows_df)} valid windows")
        return windows_df
    else:
        print("No valid windows were created")
        return pd.DataFrame()

def predict_activities(windows_df, model, scaler):
    """Use the pretrained model to predict activities for each window"""
    if windows_df.empty:
        print("No windows to predict")
        return pd.DataFrame()
    
    # Extract feature columns (exclude metadata columns)
    metadata_cols = ['window_id', 'window_start', 'window_end', 'sample_count']
    feature_cols = [col for col in windows_df.columns if col not in metadata_cols]
    
    # Extract features
    X = windows_df[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    print("Predicting activities using Random Forest model")
    predictions = model.predict(X_scaled)
    
    # Add predictions to windows dataframe
    windows_df['predicted_activity'] = predictions
    
    return windows_df

def identify_activity_segments(windows_df):
    """Group consecutive windows with the same activity into segments"""
    if windows_df.empty:
        return []
    
    segments = []
    current_activity = None
    segment_start = None
    
    # Sort by window start time
    windows_df = windows_df.sort_values('window_start')
    
    # Iterate through windows to find activity transitions
    for _, window in windows_df.iterrows():
        activity = window['predicted_activity']
        
        # Start new segment if activity changes
        if activity != current_activity:
            # Save previous segment if it exists
            if current_activity is not None:
                segments.append({
                    'activity': current_activity,
                    'start_time': segment_start,
                    'end_time': window['window_start']
                })
            
            # Start new segment
            current_activity = activity
            segment_start = window['window_start']
    
    # Add the last segment
    if current_activity is not None:
        segments.append({
            'activity': current_activity,
            'start_time': segment_start,
            'end_time': windows_df['window_end'].max()
        })
    
    return segments

def format_timestamp_as_seconds(timestamp, reference_time):
    """Convert timestamp to seconds from the reference time"""
    return timestamp - reference_time

def visualize_activity_segments(segments, reference_time, duration):
    """Create a timeline visualization of activity segments"""
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Set up colors for different activities
    colors = {
        'hammering': 'red',
        'sawing': 'blue',
        'no_work': 'green'
    }
    
    # Plot segments
    for segment in segments:
        activity = segment['activity']
        start = format_timestamp_as_seconds(segment['start_time'], reference_time)
        end = format_timestamp_as_seconds(segment['end_time'], reference_time)
        
        plt.barh(
            0, 
            width=end-start, 
            left=start, 
            height=0.5, 
            color=colors.get(activity, 'gray'),
            label=activity
        )
        
        # Add activity label in the middle of the segment
        if end - start > 3:  # Only add text if segment is wide enough
            plt.text(
                (start + end) / 2, 
                0, 
                activity,
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    # Set up axes
    plt.yticks([])
    plt.xlim(0, duration)
    plt.xlabel('Time (seconds)')
    plt.title('Activity Timeline')
    plt.tight_layout()
    
    # Save the visualization
    output_dir = os.path.join('07_real_time_evaluation', 'results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'activity_timeline.png'))
    plt.close()

def main():
    # Paths
    data_path = os.path.join('07_real_time_evaluation', 'set_for_testing_all_actions', 'IMU_R_data.csv')
    
    # Load data
    imu_data = load_imu_data(data_path)
    
    # Load model and scaler
    model, scaler = load_random_forest_model()
    
    # Reference time (first timestamp in the data)
    reference_time = imu_data['timestamp'].min()
    
    # Create windows and extract features
    windows_df = create_windows(imu_data)
    
    if windows_df.empty:
        print("No valid windows to process. Exiting.")
        return
    
    # Predict activities for each window
    windows_with_predictions = predict_activities(windows_df, model, scaler)
    
    # Group predictions into activity segments
    segments = identify_activity_segments(windows_with_predictions)
    
    # Calculate total duration
    total_duration = format_timestamp_as_seconds(imu_data['timestamp'].max(), reference_time)
    
    # Print segments as requested
    print("\nActivity Segments:")
    print("-" * 40)
    for segment in segments:
        activity = segment['activity']
        start_seconds = format_timestamp_as_seconds(segment['start_time'], reference_time)
        end_seconds = format_timestamp_as_seconds(segment['end_time'], reference_time)
        print(f"{activity}: {start_seconds:.2f} seconds to {end_seconds:.2f} seconds")
    
    # Visualize segments
    visualize_activity_segments(segments, reference_time, total_duration)
    print(f"\nActivity timeline visualization saved to 07_real_time_evaluation/results/activity_timeline.png")
    
    # Optional: Save segments to CSV
    segments_df = pd.DataFrame(segments)
    segments_df['start_seconds'] = segments_df['start_time'].apply(
        lambda x: format_timestamp_as_seconds(x, reference_time)
    )
    segments_df['end_seconds'] = segments_df['end_time'].apply(
        lambda x: format_timestamp_as_seconds(x, reference_time)
    )
    segments_df['duration'] = segments_df['end_seconds'] - segments_df['start_seconds']
    
    output_dir = os.path.join('07_real_time_evaluation', 'results')
    os.makedirs(output_dir, exist_ok=True)
    segments_df.to_csv(os.path.join(output_dir, 'activity_segments.csv'), index=False)
    print(f"Activity segments saved to 07_real_time_evaluation/results/activity_segments.csv")

if __name__ == "__main__":
    main()
