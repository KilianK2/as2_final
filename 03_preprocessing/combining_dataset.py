import os
import csv
import pandas as pd
import re

def extract_activity_and_session(folder_name):
    """Extract activity label and session ID from folder name like 'hammering_1' or 'no_work_1'"""
    # Use regex to extract the activity name and session id
    # Updated pattern to correctly capture "no_work" as the activity
    match = re.match(r'([a-z_]+)_(\d+)', folder_name)
    if match:
        activity = match.group(1)
        session_id = match.group(2)
        return activity, session_id
    return None, None

def process_data_folder(folder_path, output_file, append=False):
    """Process a data folder and append its contents to the output file with labels"""
    folder_name = os.path.basename(folder_path)
    activity, session_id = extract_activity_and_session(folder_name)
    
    print(f"Processing folder: {folder_name}")
    
    if not activity or not session_id:
        print(f"Skipping folder '{folder_path}' - can't extract activity and session ID")
        return 0
    
    print(f"Extracted activity: '{activity}', session_id: '{session_id}'")
    
    # Check for IMU_R_data.csv file
    imu_file = os.path.join(folder_path, "IMU_R_data.csv")
    if not os.path.exists(imu_file):
        print(f"Skipping folder '{folder_path}' - no IMU_R_data.csv file found")
        return 0
    
    # Read the IMU data file
    data = pd.read_csv(imu_file)
    
    # Skip header row if appending
    write_header = not append
    
    # Add activity and session_id columns
    data['activity'] = activity
    data['session_id'] = session_id
    
    # Write to output file
    data.to_csv(output_file, mode='a' if append else 'w', index=False, header=write_header)
    
    print(f"Processed {len(data)} rows from '{folder_path}'")
    return len(data)

def combine_datasets(data_root, output_file):
    """Combine all IMU datasets from data folder into one file with labels"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all folders in the data directory
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    
    # Sort folders to ensure consistent processing order
    folders.sort()
    
    total_rows = 0
    first_folder = True
    
    print(f"Found {len(folders)} data folders to process: {', '.join(folders)}")
    
    # Process each folder
    for folder in folders:
        folder_path = os.path.join(data_root, folder)
        rows_processed = process_data_folder(folder_path, output_file, append=not first_folder)
        total_rows += rows_processed
        
        if first_folder and rows_processed > 0:
            first_folder = False
    
    print(f"Processing complete. Combined {total_rows} rows into {output_file}")

if __name__ == "__main__":
    data_root = "02_data"
    output_file = "03_preprocessing/combined_raw_data.csv"
    
    print(f"Starting dataset combination process...")
    combine_datasets(data_root, output_file)
    
    # Verify the output file
    if os.path.exists(output_file):
        output_data = pd.read_csv(output_file)
        print(f"Created combined dataset with {len(output_data)} rows and {len(output_data.columns)} columns")
        print(f"Activities in dataset: {output_data['activity'].unique()}")
        print(f"Sessions in dataset: {output_data['session_id'].unique()}")
        print(f"Sample of combined data:")
        print(output_data.head())
    else:
        print(f"Error: Output file was not created")
