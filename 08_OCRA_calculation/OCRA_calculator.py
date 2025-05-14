import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, 'combined_raw_data.csv')

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"Could not find '{csv_filename}' in {script_dir}")

# Load the CSV file
df = pd.read_csv(csv_filename)

# Compute acceleration magnitude at each time point in the dataset
df['acc_mag'] = (df['AX']**2 + df['AY']**2 + df['AZ']**2) ** 0.5

# Filter for hammering and sawing
# Post-processed data contains the activity labels
hammering_df = df[df['activity'] == 'hammering']
sawing_df = df[df['activity'] == 'sawing']

# Dynamic threshold: mean_acc_mag + N*std_acc_mag for the purpose of noise filtering
N = 0.25  # N = 0.25 was selected based on empirical tuning of the visualised data peaks
hammering_mean = hammering_df['acc_mag'].mean()
hammering_std = hammering_df['acc_mag'].std()
hammering_threshold = hammering_mean + N * hammering_std

sawing_mean = sawing_df['acc_mag'].mean()
sawing_std = sawing_df['acc_mag'].std()
sawing_threshold = sawing_mean + N * sawing_std

# Biomechanical Constraint: eliminates duplicate peaks for repetitive work
min_time_between_strokes = 0.5  # Minimum feasible time between two consecutive strokes for long term repetitive work (sec). 

# Detection of hammering strokes
hammering_peaks, hammering_props = find_peaks(
    hammering_df['acc_mag'],
    height=hammering_threshold
)

# Enforcment of minimum time limit between hammering strokes
hammering_times = hammering_df['timestamp'].values[hammering_peaks]
hammering_valid = [0]
for i in range(1, len(hammering_times)):
    if (hammering_times[i] - hammering_times[hammering_valid[-1]]) >= min_time_between_strokes:
        hammering_valid.append(i)
hammering_peaks = hammering_peaks[hammering_valid]

print(f"Total number of hammering strokes: {len(hammering_peaks)}")

# Detection of sawing strokes
sawing_peaks, sawing_props = find_peaks(
    sawing_df['acc_mag'],
    height=sawing_threshold
)

# Enforcment of minimum time limit between sawing strokes
sawing_times = sawing_df['timestamp'].values[sawing_peaks]
sawing_valid = [0]
for i in range(1, len(sawing_times)):
    if (sawing_times[i] - sawing_times[sawing_valid[-1]]) >= min_time_between_strokes:
        sawing_valid.append(i)
sawing_peaks = sawing_peaks[sawing_valid]

print(f"Total number of sawing strokes (biomechanical constraints): {len(sawing_peaks)}")


# The total duration of hammering, sawing and no_work (miscellaneous) activities 
# is detected and calculated by another python module (not shown here)

# The given dataset (combined_raw_data.csv) contains 12 minutes of hammering and sawing activity each
hammering_duration = 12
sawing_duration = 12

# Calculation of number of strokes per minute
print(f"\nHammering strokes per minute: {len(hammering_peaks) / hammering_duration:.2f}")
print(f"Sawing strokes per minute: {len(sawing_peaks) / sawing_duration:.2f}")


# --- Sample OCRA Index Calculation --- #

# The exact workshift breakdown can differ from company to company.

# In the given example, the 8 hour work shift (1 hour lunch break and 1 hour non-productivity period included)
# is divided into 30-min blocks, which is broken into 3 activities: 
# 12 min hammering, 12 min sawing, and 6 min no_work (miscellaneous) 

# Number of 30-min blocks in 6 hours (8 - 1 hours for lunch and 1 hour non-productive time)
blocks = 6 * 60 // 30  # = 12

# Total minutes per activity in an 8 hour shift
t_hammering_shift = 12 * blocks  
t_sawing_shift = 12 * blocks     

# Total number of strokes detected in 12 minutes of dataset
hammering_strokes_per_min = len(hammering_peaks) / 12
sawing_strokes_per_min = len(sawing_peaks) / 12

# In the original application, the dataset will contain IMU data for a full shift (8 hours)
#Since this example dataset contains only 12 minutes of activity, we need to scale the number of strokes to a full shift
# Assuming 20% loss of productivity every 4 hours of work hammering/sawing strokes per shift are:
ATA_hammering_shift = hammering_strokes_per_min * t_hammering_shift * 0.8 * 0.8
ATA_sawing_shift = sawing_strokes_per_min * t_sawing_shift * 0.8 * 0.8

# OCRA multipliers (as before)
kf = 30 # constant reference value given by ISO 11228-3
FM_hammering = 0.85 # As per ISO 11228-3 Section C.4.3; Target group defined "hammering" activity to be a "very weak" usage of force according to CR-10 Borg Scale
FM_sawing = 0.65 # As per ISO 11228-3 Section C.4.3; Target group defined "sawing" activity to be a " weak" usage of force according to CR-10 Borg Scale
PM_hammering = 1 # As per ISO 11228-3 Section C.4.4; Awkward posture and/or movement takes place less than 1/3 of the time for the "hammering" activity 
PM_sawing = 0.5 # As per ISO 11228-3 Section C.4.4; Saw is held with a "Hook Grip" all th time during "sawing" activity
Rem_hammering = 0.7 # As per ISO 11228-3 Section C.4.4; Cycle is shorter than 15 seconds
Rem_sawing = 0.7 # As per ISO 11228-3 Section C.4.4; Cycle is shorter than 15 seconds
AM_hammering = 0.8 # As per ISO 11228-3 Section C.4.6; Hammering gesture implies countershock
AM_sawing = 1 # Defined as per ISO 11228-3 Section C.4.6;

# ISO 11228-3 C.4.8: RcM = 1 if recovery/work ratio > 0.2
# Current ratio = 144/480 = 0.3 → RcM = 1
Rcm = 1  

# Duration multiplier for the shift (tM): for a full shift, tM = 1
tM = 1

# Calculate RTA for each activity (for full shift)
RTA_hammering = kf * FM_hammering * PM_hammering * Rem_hammering * AM_hammering * t_hammering_shift
RTA_sawing = kf * FM_sawing * PM_sawing * Rem_sawing * AM_sawing * t_sawing_shift

# Total RTA (apply Rcm and tM after summing)
RTA_total = (RTA_hammering + RTA_sawing) * (Rcm * tM)

# OCRA Index for full shift
OCRA_total = (ATA_hammering_shift + ATA_sawing_shift) / RTA_total

print("\n--- OCRA Index Calculation for 8-hour shift ---")
print(f"Hammering: ATA = {ATA_hammering_shift:.0f}, RTA = {RTA_hammering:.2f}")
print(f"Sawing:    ATA = {ATA_sawing_shift:.0f}, RTA = {RTA_sawing:.2f}")
print(f"Total RTA (with Rcm and tM): {RTA_total:.2f}")
print(f"Combined OCRA Index: {OCRA_total:.2f}")