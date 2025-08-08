# AffectiveLens: Hugging Face Model Uploader
#
# This script uploads all trained model files (.pkl) from a local directory
# to a specified repository on the Hugging Face Hub.
#
# Author: Sanyam Sanjay Sharma
#
# INSTRUCTIONS:
# 1. AUTHENTICATE FIRST: Run 'huggingface-cli login' in your terminal and paste a 'write' token.
# 2. CONFIGURE: Ensure the variables in the Configuration section are correct.
# 3. RUN: Execute this script from your project's root directory: `python upload_models_to_hf.py`

from huggingface_hub import HfApi, create_repo
import os

# --- 1. Configuration ---

# The name of your model repository on the Hugging Face Hub.
# This should be in the format: "your-username/your-repo-name"
MODEL_REPO_ON_HUB = "<your database repository name as per the above instructions>"

# The local folder where your trained models (.pkl files) are saved.
LOCAL_MODELS_FOLDER = "./notebooks/trained_models"


def upload_models_to_hub():
    """
    Uploads the entire contents of a local folder to a Hugging Face model repository.
    """
    print("--- Starting Hugging Face Hub Model Upload Process ---")

    # --- 2. Verify Local Models Folder ---
    if not os.path.exists(LOCAL_MODELS_FOLDER) or not os.listdir(LOCAL_MODELS_FOLDER):
        print("\n--- ERROR ---")
        print(f"The local models folder is empty or was not found at: '{LOCAL_MODELS_FOLDER}'")
        print("Please ensure you have trained and saved your models first.")
        return
        
    print(f"Found local models folder at: '{LOCAL_MODELS_FOLDER}'")
    
    # --- 3. Initialize the API and Create the Repo ---
    api = HfApi()
    
    print(f"\nCreating or getting repository '{MODEL_REPO_ON_HUB}' on the Hub...")
    try:
        # Create a new repository on the Hub.
        # `repo_type='model'` is the default, but we specify it for clarity.
        # `exist_ok=True` means it won't fail if the repo already exists.
        repo_url = create_repo(
            repo_id=MODEL_REPO_ON_HUB,
            exist_ok=True,
        )
        print(f"Repository is ready at: {repo_url}")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Could not create the repository on the Hub. Error: {e}")
        print("Please ensure you have authenticated correctly and have write permissions.")
        return

    # --- 4. Upload the Folder Contents ---
    try:
        print(f"\nUploading contents of local folder '{LOCAL_MODELS_FOLDER}'...")
        
        # The `upload_folder` function is a powerful utility that handles everything:
        # It clones the repo, copies your files, and pushes them in one command.
        api.upload_folder(
            folder_path=LOCAL_MODELS_FOLDER,
            repo_id=MODEL_REPO_ON_HUB,
            repo_type="model", # Specify the type as 'model'
        )

        print("\n--- UPLOAD COMPLETE ---")
        print("Your models are now available on the Hugging Face Hub at:")
        print(f"https://huggingface.co/{MODEL_REPO_ON_HUB}")

    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED DURING UPLOAD ---")
        print(f"Error details: {e}")


if __name__ == "__main__":
    upload_models_to_hub()
