# AffectiveLens: Automated Data and Embeddings Downloader
#
# This script uses the standard Kaggle command-line interface (CLI) to download
# the complete dataset, which includes directories of pre-computed embeddings.
#
# Author: Sanyam Sanjay Sharma
#
# Instructions:
# 1. Make sure you have run 'pip install kaggle' and have your `kaggle.json` API token set up.
# 2. Run this script from your project's root directory: `python download_data.py`

import os
import subprocess
import sys

# --- Configuration ---
# The user and name of the dataset on Kaggle.
DATASET_SLUG = "kianhutchinson/mentalheathdatabase" 

# The local directory where the dataset's contents will be downloaded and unzipped.
# This will create the 'MentalTrain' and 'MentalTest' folders inside this path.
LOCAL_DATA_PATH = "./notebooks/data/embeddings/"

def download_and_save_embeddings():
    """
    Downloads and unzips a dataset from Kaggle into a structured local directory.
    """
    print("--- Starting Automated Embedding Download ---")
    
    # Create the local directory if it doesn't exist
    print(f"Ensuring local save directory exists: {LOCAL_DATA_PATH}")
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
    
    try:
        # --- Construct the Kaggle CLI command ---
        # This is the standard way to download a full dataset.
        command = [
            "kaggle",
            "datasets",
            "download",
            "-d", DATASET_SLUG,  # Specify the dataset
            "-p", LOCAL_DATA_PATH, # Specify the path to download to
            "--unzip"              # Automatically unzip the contents
        ]
        
        print(f"\nExecuting Kaggle command: {' '.join(command)}")
        
        # --- Run the command ---
        # Using subprocess.run is a robust way to execute shell commands from Python.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("\n--- Kaggle CLI Output ---")
        print(result.stdout)
        
        print(f"\nSuccessfully downloaded and unzipped dataset to: {LOCAL_DATA_PATH}")
        print("The 'MentalTrain' and 'MentalTest' folders should now be available inside.")

    except FileNotFoundError:
        print("\n--- AN ERROR OCCURRED ---")
        print("Error: The 'kaggle' command was not found.")
        print("Please ensure the Kaggle library is installed by running: 'pip install kaggle'")

    except subprocess.CalledProcessError as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print("The Kaggle command failed to execute.")
        print(f"Error details: {e.stderr}")
        print("\nPlease ensure the following:")
        print("1. Your Kaggle API token ('kaggle.json') is correctly configured.")
        print(f"2. The dataset '{DATASET_SLUG}' exists and is public.")
        
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error details: {e}")

    print("\n--- Download process finished. ---")

if __name__ == "__main__":
    # This block runs when the script is executed directly
    download_and_save_embeddings()
