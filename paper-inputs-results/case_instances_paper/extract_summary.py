# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:59:21 2025

@author: erdidasdemir
"""

import os
import pandas as pd

# Define the root directory (where this script is located)
root_dir = os.getcwd()

# Initialize an empty list to store DataFrames
data_frames = []

# Loop over the 7 main folders
for folder in sorted(os.listdir(root_dir)):  # Sorting to maintain order
    folder_path = os.path.join(root_dir, folder)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        ccr_results_path = os.path.join(folder_path, "ccr-results")

        # Check if "ccr-results" folder exists
        if os.path.exists(ccr_results_path) and os.path.isdir(ccr_results_path):
            # Iterate over 6 Excel files
            for file in sorted(os.listdir(ccr_results_path)):  # Sorting for consistency
                if file.endswith(".xlsx"):
                    file_path = os.path.join(ccr_results_path, file)
                    
                    # Read the global_results sheet
                    df = pd.read_excel(file_path, sheet_name="global_results")

                    # Ensure there's only one row in the file
                    if len(df) == 1:
                        # Add a column to indicate the source file
                        df["Source_File"] = file
                        df["Folder"] = folder
                        data_frames.append(df)

# Combine all collected data
if data_frames:
    final_df = pd.concat(data_frames, ignore_index=True)

    # Save as an Excel file
    output_file = "compiled_results.xlsx"
    final_df.to_excel(output_file, index=False)
    print(f"Data saved successfully to {output_file}")
else:
    print("No data was found in the specified files.")
