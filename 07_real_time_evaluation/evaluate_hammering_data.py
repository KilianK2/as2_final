import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict

# Add parent directories to path to import feature extraction code
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '04_feature extraction'))
from feature_extraction import extract_features

# Import functions from the main evaluation script
from real_time_evaluation import (
    load_random_forest_model,
    load_imu_data,
    create_windows,
    predict_activities,
    identify_activity_segments,
    format_timestamp_as_seconds
)

def visualize_activity_segments(segments, reference_time, duration, output_dir):
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
    plt.title('Hammering Test - Activity Timeline')
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hammering_activity_timeline.png'))
    plt.close()

def main():
    # Set up output directory for hammering test results
    output_dir = os.path.join('07_real_time_evaluation', 'results_hammering_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for IMU data in the hammering test directory
    test_dir = os.path.join('07_real_time_evaluation', 'testing_hammering')
    data_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    if not data_files:
        print(f"No CSV files found in {test_dir}")
        return
    
    print(f"Found {len(data_files)} data files: {', '.join(data_files)}")
    
    # Process the first CSV file (or you could loop through all files)
    data_path = os.path.join(test_dir, data_files[0])
    
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
    print("\nHammering Test - Activity Segments:")
    print("-" * 50)
    for segment in segments:
        activity = segment['activity']
        start_seconds = format_timestamp_as_seconds(segment['start_time'], reference_time)
        end_seconds = format_timestamp_as_seconds(segment['end_time'], reference_time)
        print(f"{activity}: {start_seconds:.2f} seconds to {end_seconds:.2f} seconds")
    
    # Visualize segments
    visualize_activity_segments(segments, reference_time, total_duration, output_dir)
    print(f"\nActivity timeline visualization saved to {os.path.join(output_dir, 'hammering_activity_timeline.png')}")
    
    # Save segments to CSV
    segments_df = pd.DataFrame(segments)
    segments_df['start_seconds'] = segments_df['start_time'].apply(
        lambda x: format_timestamp_as_seconds(x, reference_time)
    )
    segments_df['end_seconds'] = segments_df['end_time'].apply(
        lambda x: format_timestamp_as_seconds(x, reference_time)
    )
    segments_df['duration'] = segments_df['end_seconds'] - segments_df['start_seconds']
    
    segments_df.to_csv(os.path.join(output_dir, 'hammering_activity_segments.csv'), index=False)
    print(f"Activity segments saved to {os.path.join(output_dir, 'hammering_activity_segments.csv')}")
    
    # Additional analysis: Calculate percentage of time spent in each activity
    total_seconds = segments_df['duration'].sum()
    activity_percentages = segments_df.groupby('activity')['duration'].sum() / total_seconds * 100
    
    print("\nTime spent in each activity:")
    for activity, percentage in activity_percentages.items():
        print(f"{activity}: {percentage:.1f}%")
    
    # Define colors for activities - fix for the error
    colors = {
        'hammering': 'red',
        'sawing': 'blue',
        'no_work': 'green'
    }
    
    # Create a pie chart of activity distribution
    plt.figure(figsize=(8, 8))
    plt.pie(
        activity_percentages, 
        labels=activity_percentages.index,
        autopct='%1.1f%%',
        colors=[colors.get(a, 'gray') for a in activity_percentages.index],
        startangle=90
    )
    plt.title('Distribution of Activities in Hammering Test')
    plt.savefig(os.path.join(output_dir, 'hammering_activity_distribution.png'))
    plt.close()

if __name__ == "__main__":
    main() 