import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
import joblib
from collections import deque
from datetime import datetime

# Add parent directory to path to import feature extraction
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '04_feature extraction'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_splitting'))

# Import feature extraction module
from feature_extraction import extract_features

class RealTimeEvaluator:
    def __init__(self, model_path, window_size=3.0, step_size=0.5):
        """
        Initialize the real-time evaluator.
        
        Args:
            model_path: Path to the trained model
            window_size: Size of the sliding window in seconds
            step_size: Step size between windows in seconds
        """
        self.window_size = window_size
        self.step_size = step_size
        
        # Load the trained model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model loaded successfully!")
        
        # Initialize figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('Activity Recognition')
        
        # Class colors for visualization
        self.class_colors = {
            'hammering': 'red',
            'sawing': 'blue',
            'no_work': 'green'
        }
        
        # Store predictions over time
        self.timestamps = []
        self.predictions = []
        self.window_data = []
        
        # For smoothing predictions (majority voting over recent windows)
        self.recent_predictions = deque(maxlen=5)
        
    def load_data(self, data_file):
        """Load IMU data from file"""
        print(f"Loading IMU data from {data_file}...")
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} samples")
        
        # Check if the data has timestamp column
        if 'timestamp' not in data.columns:
            print("Warning: No timestamp column found. Creating artificial timestamps...")
            # Create artificial timestamps at 50Hz
            start_time = time.time()
            data['timestamp'] = [start_time + i/50.0 for i in range(len(data))]
        
        return data
    
    def process_window(self, window_data):
        """Process a window of data and make a prediction"""
        # Extract features from the window
        features = extract_features(window_data)
        
        # Convert to DataFrame with matching column order
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        
        # Add to recent predictions for smoothing
        self.recent_predictions.append(prediction)
        
        # Get majority vote from recent predictions
        if len(self.recent_predictions) > 0:
            prediction_counts = {}
            for pred in self.recent_predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            # Get the most common prediction
            smoothed_prediction = max(prediction_counts.items(), key=lambda x: x[1])[0]
            return smoothed_prediction
        
        return prediction
    
    def create_windows(self, data):
        """Create sliding windows from the data and make predictions"""
        # Sort data by timestamp to ensure correct window creation
        data = data.sort_values('timestamp')
        
        # Get timestamps
        timestamps = data['timestamp'].values
        
        if len(timestamps) == 0:
            print("Error: No data found")
            return
        
        # Calculate window parameters
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_duration = end_time - start_time
        
        print(f"Data spans {total_duration:.2f} seconds")
        
        # Normalize timestamps relative to start_time for better visualization
        normalized_timestamps = timestamps - start_time
        data['normalized_timestamp'] = normalized_timestamps
        
        # Create windows
        window_starts = np.arange(start_time, end_time - self.window_size + 0.1, self.step_size)
        
        print(f"Creating {len(window_starts)} windows...")
        
        # Process each window
        for window_start in window_starts:
            window_end = window_start + self.window_size
            
            # Get data in this time window
            window_data = data[(data['timestamp'] >= window_start) & 
                              (data['timestamp'] < window_end)].copy()
            
            # Only process windows with sufficient data
            if len(window_data) < 10:  # Minimum samples for a useful window
                continue
                
            # Process window and get prediction
            normalized_start = window_start - start_time
            prediction = self.process_window(window_data)
            
            # Store results
            self.timestamps.append(normalized_start)
            self.predictions.append(prediction)
            self.window_data.append({
                'start': normalized_start,
                'end': window_end - start_time,
                'prediction': prediction
            })
    
    def visualize_predictions(self):
        """Create a static visualization of predictions over time"""
        if not self.window_data:
            print("No predictions to visualize")
            return
        
        # Clear the axis
        self.ax.clear()
        
        # Plot the predictions as colored segments
        for i, window in enumerate(self.window_data):
            if i == 0:
                continue  # Skip the first window
                
            # Get the current and previous predictions
            current_pred = window['prediction']
            prev_pred = self.window_data[i-1]['prediction']
            
            # Only create a new segment if the prediction changes
            if current_pred != prev_pred or i == 1:
                start = window['start']
                end = window['end']
                y_pos = 0.5
                height = 1.0
                color = self.class_colors.get(current_pred, 'gray')
                
                # Add colored rectangle
                rect = Rectangle((start, y_pos - height/2), 
                                 end - start, 
                                 height, 
                                 facecolor=color, 
                                 alpha=0.7)
                self.ax.add_patch(rect)
                
                # Add text label
                self.ax.text(start + (end - start) / 2, 
                             y_pos, 
                             current_pred, 
                             ha='center', 
                             va='center', 
                             color='white', 
                             fontweight='bold')
        
        # Set up axis and labels
        self.ax.set_xlim(0, max([w['end'] for w in self.window_data]))
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Activity Recognition Over Time')
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_yticks([])
        
        # Add legend
        patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in self.class_colors.values()]
        self.ax.legend(patches, self.class_colors.keys(), loc='upper right')
        
        # Display the plot
        plt.tight_layout()
        plt.savefig('results/activity_timeline.png')
        plt.show()

    def create_animated_visualization(self):
        """Create an animated visualization of predictions over time"""
        if not self.window_data:
            print("No predictions to visualize")
            return
            
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.canvas.manager.set_window_title('Activity Recognition (Animated)')
        
        # Maximum time for x-axis
        max_time = max([w['end'] for w in self.window_data])
        
        # Animation update function
        def update(frame):
            # Clear axis
            ax.clear()
            
            # Find all windows up to the current frame time
            current_time = frame * max_time / 100  # Convert frame to time
            
            # Keep track of the latest prediction for each time segment
            segments = []
            prev_pred = None
            segment_start = 0
            
            for window in self.window_data:
                if window['start'] <= current_time:
                    # If prediction changes, close the current segment
                    if prev_pred is not None and window['prediction'] != prev_pred:
                        segments.append({
                            'start': segment_start,
                            'end': window['start'],
                            'prediction': prev_pred
                        })
                        segment_start = window['start']
                    
                    prev_pred = window['prediction']
            
            # Add the last segment
            if prev_pred is not None:
                segments.append({
                    'start': segment_start,
                    'end': current_time,
                    'prediction': prev_pred
                })
            
            # Plot the segments
            for segment in segments:
                start = segment['start']
                end = segment['end']
                y_pos = 0.5
                height = 1.0
                color = self.class_colors.get(segment['prediction'], 'gray')
                
                # Add colored rectangle
                rect = Rectangle((start, y_pos - height/2), 
                                end - start, 
                                height, 
                                facecolor=color, 
                                alpha=0.7)
                ax.add_patch(rect)
                
                # Add text label if segment is wide enough
                if end - start > max_time * 0.03:
                    ax.text(start + (end - start) / 2, 
                            y_pos, 
                            segment['prediction'], 
                            ha='center', 
                            va='center', 
                            color='white', 
                            fontweight='bold')
            
            # Set up axis and labels
            ax.set_xlim(0, max_time)
            ax.set_ylim(0, 1)
            ax.set_title(f'Activity Recognition Over Time (Current Time: {current_time:.1f}s)')
            ax.set_xlabel('Time (seconds)')
            ax.set_yticks([])
            
            # Add current time line
            ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
            
            # Add legend
            patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in self.class_colors.values()]
            ax.legend(patches, self.class_colors.keys(), loc='upper right')
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=101, interval=50, blit=False)
        
        # Save animation
        ani.save('results/activity_timeline.gif', writer='pillow', fps=20)
        
        # Show the animation
        plt.tight_layout()
        plt.show()

def main():
    # Project paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "results", "random_forest_model.joblib")
    test_data_path = os.path.join(project_root, "07_real_time_evaluation", 
                                "set_for_testing_all_actions", "IMU_R_data.csv")
    
    # Make sure the results directory exists
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    
    # Check if the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Check if the test data exists
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at: {test_data_path}")
    
    # Initialize evaluator
    evaluator = RealTimeEvaluator(model_path, window_size=2.0, step_size=0.5)
    
    # Load test data
    data = evaluator.load_data(test_data_path)
    
    # Process the data and make predictions
    evaluator.create_windows(data)
    
    # Visualize predictions
    evaluator.visualize_predictions()
    
    # Create animated visualization
    evaluator.create_animated_visualization()

if __name__ == "__main__":
    main()
