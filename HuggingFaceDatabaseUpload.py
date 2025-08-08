# AffectiveLens: Hugging Face Full Data Folder Uploader
#
# This script uploads an entire local directory (like your 'data' folder)
# to a new or existing dataset repository on the Hugging Face Hub.
#
# Author: Sanyam Sanjay Sharma
#
# INSTRUCTIONS:
# 1. AUTHENTICATE FIRST: Run 'huggingface-cli login' in your terminal and paste a 'write' token.
# 2. CONFIGURE: Update the variables in the Configuration section below.
# 3. RUN: Execute this script from your project's root directory: `python upload_folder_to_hf.py`

from huggingface_hub import HfApi, create_repo
import os
import shutil

# --- 1. Configuration ---

# The name of your dataset repository on the Hugging Face Hub.
# This should be in the format: "your-username/your-dataset-name"
DATASET_NAME_ON_HUB = "<your database repository name as per the above instructions>"

# The local folder you want to upload.
# This script will upload the entire contents of this folder.
LOCAL_FOLDER_TO_UPLOAD = "./notebooks/data"

# A temporary folder to work in. This will be created and then deleted.
TEMP_CLONE_FOLDER = "./temp_hf_clone"


def upload_full_folder_to_hub():
    """
    Uploads the entire contents of a local folder to a Hugging Face dataset repository.
    """
    print("--- Starting Hugging Face Hub Folder Upload Process ---")
    
    # --- 2. Initialize the API and Create the Repo ---
    api = HfApi()
    
    print(f"\nCreating or getting repository '{DATASET_NAME_ON_HUB}' on the Hub...")
    try:
        # Create a new dataset repository on the Hub.
        # `repo_type='dataset'` is crucial.
        # `exist_ok=True` means it won't fail if the repo already exists.
        repo_url = create_repo(
            repo_id=DATASET_NAME_ON_HUB,
            repo_type="dataset",
            exist_ok=True,
        )
        print(f"Repository is ready at: {repo_url}")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Could not create the repository on the Hub. Error: {e}")
        print("Please ensure you have authenticated correctly and have write permissions.")
        return

    # --- 3. Upload the Folder Contents ---
    try:
        print(f"\nUploading contents of local folder '{LOCAL_FOLDER_TO_UPLOAD}'...")
        
        # The `upload_folder` function is a powerful utility that handles everything:
        # It clones the repo, copies your files, and pushes them in one command.
        api.upload_folder(
            folder_path=LOCAL_FOLDER_TO_UPLOAD,
            repo_id=DATASET_NAME_ON_HUB,
            repo_type="dataset",
        )

        print("\n--- UPLOAD COMPLETE ---")
        print("Your data folder is now available on the Hugging Face Hub at:")
        print(f"https://huggingface.co/datasets/{DATASET_NAME_ON_HUB}")

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print(f"The local folder to upload was not found at: '{LOCAL_FOLDER_TO_UPLOAD}'")
        print("Please ensure the path is correct and you are running the script from your project's root directory.")
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED DURING UPLOAD ---")
        print(f"Error details: {e}")


if __name__ == "__main__":
    upload_full_folder_to_hub()
