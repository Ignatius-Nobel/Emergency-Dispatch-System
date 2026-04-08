import os
from huggingface_hub import HfApi

def upload_space():
    print("Uploading to Hugging Face...")
    api = HfApi()
    api.upload_folder(
        folder_path=".",
        repo_id="ignatius-nobel/Emergency-Dispatch-System",
        repo_type="space",
        ignore_patterns=[
            ".venv/*", ".venv", 
            "venv/*", "venv", 
            ".git/*", ".git", 
            "__pycache__/*",
            "*.egg-info/*", "*.egg-info"
        ]
    )
    print("Upload complete!")

if __name__ == "__main__":
    upload_space()
