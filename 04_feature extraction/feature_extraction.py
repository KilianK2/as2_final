import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def load_combined_data(file_path):
    """Load the combined IMU data from CSV"""
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} samples with columns: {', '.join(data.columns)}")
    return data

def create_windows(data, window_size_seconds, overlap_percent):
    """Split data into windows with specified overlap"""
    # Sort data by timestamp to ensure correct window creation
    data = data.sort_values('timestamp')
    
    # Group data by activity and session_id
    grouped = data.groupby(['activity', 'session_id'])
    
    all_windows = []
    
    for (activity, session_id), group in grouped:
        print(f"Processing activity: {activity}, session: {session_id} with {len(group)} samples")
        
        # Get timestamps for this activity/session
        timestamps = group['timestamp'].values
        
        if len(timestamps) == 0:
            continue
            
        # Calculate window parameters
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_duration = end_time - start_time
        
        # Convert window size to seconds
        window_size = window_size_seconds  # in seconds
        step_size = window_size * (1 - overlap_percent / 100)  # in seconds
        
        print(f"  Window size: {window_size}s, Step size: {step_size}s, Total duration: {total_duration:.2f}s")
        
        # Create windows
        window_starts = np.arange(start_time, end_time - window_size + 0.1, step_size)
        
        print(f"  Creating {len(window_starts)} windows...")
        
        # Process each window
        for i, window_start in enumerate(window_starts):
            window_end = window_start + window_size
            
            # Get data in this time window
            window_data = group[(group['timestamp'] >= window_start) & 
                               (group['timestamp'] < window_end)]
            
            # Only process windows with sufficient data
            if len(window_data) < 10:  # Minimum samples for a useful window
                continue
                
            # Extract features for this window
            features = extract_features(window_data)
            
            # Add metadata
            features['activity'] = activity
            features['session_id'] = session_id
            features['window_id'] = i
            features['window_start'] = window_start
            features['window_end'] = window_end
            features['sample_count'] = len(window_data)
            
            all_windows.append(features)
    
    # Combine all windows into a single DataFrame
    if all_windows:
        windows_df = pd.DataFrame(all_windows)
        print(f"Created {len(windows_df)} windows across all activities and sessions")
        return windows_df
    else:
        print("No valid windows were created")
        return pd.DataFrame()

def extract_features(window_data):
    """Extract statistical features from a window of IMU data"""
    features = {}
    
    # Names of gyroscope and accelerometer columns
    gyro_cols = ['GX', 'GY', 'GZ']
    acc_cols = ['AX', 'AY', 'AZ']
    
    # Get sample rate for frequency analysis
    if len(window_data) > 1:
        timestamps = window_data['timestamp'].values
        # Estimate sample rate from average time difference
        sample_rate = 1 / np.mean(np.diff(timestamps))
    else:
        # Default to 50Hz if we can't calculate
        sample_rate = 50.0
    
    # 1-5. Basic statistical features
    # Calculate features for gyroscope data
    for i, axis in enumerate(['X', 'Y', 'Z']):
        gyro_data = window_data[gyro_cols[i]].values
        acc_data = window_data[acc_cols[i]].values
        
        # 1. Mean
        features[f'GYRO_MEAN_{axis}'] = np.mean(gyro_data)
        features[f'ACC_MEAN_{axis}'] = np.mean(acc_data)
        
        # 2. Standard Deviation
        features[f'GYRO_STD_{axis}'] = np.std(gyro_data)
        features[f'ACC_STD_{axis}'] = np.std(acc_data)
        
        # 3. Min
        features[f'GYRO_MIN_{axis}'] = np.min(gyro_data)
        features[f'ACC_MIN_{axis}'] = np.min(acc_data)
        
        # 4. Max
        features[f'GYRO_MAX_{axis}'] = np.max(gyro_data)
        features[f'ACC_MAX_{axis}'] = np.max(acc_data)
        
        # 5. Range (Max - Min)
        features[f'GYRO_RANGE_{axis}'] = features[f'GYRO_MAX_{axis}'] - features[f'GYRO_MIN_{axis}']
        features[f'ACC_RANGE_{axis}'] = features[f'ACC_MAX_{axis}'] - features[f'ACC_MIN_{axis}']
        
        # 6. Root Mean Square (RMS)
        features[f'GYRO_RMS_{axis}'] = np.sqrt(np.mean(np.square(gyro_data)))
        features[f'ACC_RMS_{axis}'] = np.sqrt(np.mean(np.square(acc_data)))
        
        # 8. Mean Absolute Deviation (MAD)
        features[f'GYRO_MAD_{axis}'] = np.mean(np.abs(gyro_data - features[f'GYRO_MEAN_{axis}']))
        features[f'ACC_MAD_{axis}'] = np.mean(np.abs(acc_data - features[f'ACC_MEAN_{axis}']))
        
        # 15. Peak Count - number of peaks in the signal
        # Use a prominence threshold based on the signal amplitude
        gyro_threshold = max(0.1 * features[f'GYRO_RANGE_{axis}'], 0.5)  # At least 0.5
        acc_threshold = max(0.1 * features[f'ACC_RANGE_{axis}'], 0.01)   # At least 0.01
        
        gyro_peaks, _ = find_peaks(gyro_data, prominence=gyro_threshold)
        acc_peaks, _ = find_peaks(acc_data, prominence=acc_threshold)
        
        features[f'GYRO_PEAK_COUNT_{axis}'] = len(gyro_peaks)
        features[f'ACC_PEAK_COUNT_{axis}'] = len(acc_peaks)
        
        # 12-14. Frequency Domain Features
        if len(gyro_data) > 4:  # Need enough data points for FFT
            # Calculate FFT
            gyro_fft = fft(gyro_data)
            acc_fft = fft(acc_data)
            
            # Calculate frequency bins
            freqs = fftfreq(len(gyro_data), 1/sample_rate)
            
            # Only use positive frequencies (first half)
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            gyro_fft = gyro_fft[pos_mask]
            acc_fft = acc_fft[pos_mask]
            
            # Calculate magnitudes of FFT coefficients
            gyro_fft_mag = np.abs(gyro_fft)
            acc_fft_mag = np.abs(acc_fft)
            
            # 12. Dominant Frequency (frequency with highest magnitude)
            if len(freqs) > 0 and len(gyro_fft_mag) > 0:
                gyro_dom_freq_idx = np.argmax(gyro_fft_mag)
                acc_dom_freq_idx = np.argmax(acc_fft_mag)
                
                features[f'GYRO_DOM_FREQ_{axis}'] = freqs[gyro_dom_freq_idx]
                features[f'ACC_DOM_FREQ_{axis}'] = freqs[acc_dom_freq_idx]
                
                # 13. Dominant Frequency Magnitude
                features[f'GYRO_DOM_FREQ_MAG_{axis}'] = gyro_fft_mag[gyro_dom_freq_idx]
                features[f'ACC_DOM_FREQ_MAG_{axis}'] = acc_fft_mag[acc_dom_freq_idx]
                
                # 14. Spectral Energy (sum of squared magnitudes)
                features[f'GYRO_SPEC_ENERGY_{axis}'] = np.sum(np.square(gyro_fft_mag)) / len(gyro_fft_mag)
                features[f'ACC_SPEC_ENERGY_{axis}'] = np.sum(np.square(acc_fft_mag)) / len(acc_fft_mag)
            else:
                # Default values if we couldn't calculate
                features[f'GYRO_DOM_FREQ_{axis}'] = 0
                features[f'ACC_DOM_FREQ_{axis}'] = 0
                features[f'GYRO_DOM_FREQ_MAG_{axis}'] = 0
                features[f'ACC_DOM_FREQ_MAG_{axis}'] = 0
                features[f'GYRO_SPEC_ENERGY_{axis}'] = 0
                features[f'ACC_SPEC_ENERGY_{axis}'] = 0
        else:
            # Default values if we don't have enough data
            features[f'GYRO_DOM_FREQ_{axis}'] = 0
            features[f'ACC_DOM_FREQ_{axis}'] = 0
            features[f'GYRO_DOM_FREQ_MAG_{axis}'] = 0
            features[f'ACC_DOM_FREQ_MAG_{axis}'] = 0
            features[f'GYRO_SPEC_ENERGY_{axis}'] = 0
            features[f'ACC_SPEC_ENERGY_{axis}'] = 0
    
    # 7. Signal Magnitude Area (SMA) - sum of absolute values across all axes
    gyro_x = window_data['GX'].values
    gyro_y = window_data['GY'].values
    gyro_z = window_data['GZ'].values
    
    acc_x = window_data['AX'].values
    acc_y = window_data['AY'].values
    acc_z = window_data['AZ'].values
    
    features['GYRO_SMA'] = np.mean(np.abs(gyro_x) + np.abs(gyro_y) + np.abs(gyro_z))
    features['ACC_SMA'] = np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
    
    # 9-11. Correlation Between Axes
    # Gyroscope correlations
    features['GYRO_CORR_X_Y'] = np.corrcoef(gyro_x, gyro_y)[0, 1] if len(gyro_x) > 1 else 0
    features['GYRO_CORR_Y_Z'] = np.corrcoef(gyro_y, gyro_z)[0, 1] if len(gyro_y) > 1 else 0
    features['GYRO_CORR_X_Z'] = np.corrcoef(gyro_x, gyro_z)[0, 1] if len(gyro_x) > 1 else 0
    
    # Accelerometer correlations
    features['ACC_CORR_X_Y'] = np.corrcoef(acc_x, acc_y)[0, 1] if len(acc_x) > 1 else 0
    features['ACC_CORR_Y_Z'] = np.corrcoef(acc_y, acc_z)[0, 1] if len(acc_y) > 1 else 0
    features['ACC_CORR_X_Z'] = np.corrcoef(acc_x, acc_z)[0, 1] if len(acc_x) > 1 else 0
    
    # 16. Y-to-X Ratio
    # Avoid division by zero
    gyro_x_mean_abs = np.mean(np.abs(gyro_x))
    acc_x_mean_abs = np.mean(np.abs(acc_x))
    
    features['GYRO_Y_X_RATIO'] = np.mean(np.abs(gyro_y)) / gyro_x_mean_abs if gyro_x_mean_abs > 0 else 0
    features['ACC_Y_X_RATIO'] = np.mean(np.abs(acc_y)) / acc_x_mean_abs if acc_x_mean_abs > 0 else 0
    
    return features

def save_features(features_df, output_path):
    """Save the extracted features to CSV"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    features_df.to_csv(output_path, index=False)
    print(f"Saved {len(features_df)} feature windows to {output_path}")
    
    # Print some statistics about the saved data
    print("\nFeature Dataset Statistics:")
    print(f"- Total windows: {len(features_df)}")
    print(f"- Total features per window: {len(features_df.columns) - 6}")  # Subtract metadata columns
    
    # Activities and their counts
    activity_counts = features_df['activity'].value_counts()
    print("\nActivity distribution:")
    for activity, count in activity_counts.items():
        print(f"- {activity}: {count} windows ({count/len(features_df)*100:.1f}%)")
    
    # Session counts
    session_counts = features_df['session_id'].value_counts()
    print("\nSession distribution:")
    for session, count in session_counts.items():
        print(f"- Session {session}: {count} windows")

def main():
    # Paths
    input_path = os.path.join("03_preprocessing", "combined_raw_data.csv")
    output_path = os.path.join("04_feature extraction", "data_by_features_per_window.csv")
    
    # Parameters
    window_size = 2.0  # seconds
    overlap = 50.0  # percent
    
    print(f"Starting feature extraction with {window_size}s windows and {overlap}% overlap")
    
    # Load data
    data = load_combined_data(input_path)
    
    # Create windows and extract features
    features_df = create_windows(data, window_size, overlap)
    
    # Save features
    if not features_df.empty:
        save_features(features_df, output_path)
        print("\nFeature extraction completed successfully!")
    else:
        print("\nFeature extraction failed: No features were generated.")

if __name__ == "__main__":
    main()
