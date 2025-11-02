"""
Dataset Download and Preparation Script for FER2013
This script downloads the FER2013 dataset from Kaggle and prepares it for training.
"""

import os
import pandas as pd
import numpy as np
import kaggle
from PIL import Image
import zipfile

def setup_kaggle_credentials():
    """
    Instructions for setting up Kaggle API credentials.
    """
    print("=== KAGGLE API SETUP INSTRUCTIONS ===")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads kaggle.json file")
    print("5. Move kaggle.json to ~/.kaggle/ directory")
    print("6. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("7. Then run this script again")
    print("=====================================")

def download_fer2013_dataset():
    """
    Download FER2013 dataset from Kaggle
    """
    try:
        print("Downloading FER2013 dataset from Kaggle...")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'msambare/fer2013', 
            path='data/', 
            unzip=True
        )
        
        print("Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure Kaggle API is properly configured.")
        setup_kaggle_credentials()
        return False

def prepare_dataset():
    """
    Prepare the dataset for training by organizing images into folders
    """
    try:
        # Check if dataset exists
        data_path = "data/fer2013"
        if not os.path.exists(data_path):
            print("Dataset not found. Please download first.")
            return False
        
        print("Preparing dataset structure...")
        
        # Create directories for each emotion
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        for split in ['train', 'test']:
            for emotion in emotions:
                os.makedirs(f"data/prepared/{split}/{emotion}", exist_ok=True)
        
        print("Dataset structure prepared!")
        print("Available emotions:", emotions)
        
        # Count images in each category
        for split in ['train', 'test']:
            split_path = os.path.join(data_path, split)
            if os.path.exists(split_path):
                total_images = 0
                for emotion in emotions:
                    emotion_path = os.path.join(split_path, emotion)
                    if os.path.exists(emotion_path):
                        count = len(os.listdir(emotion_path))
                        print(f"{split} - {emotion}: {count} images")
                        total_images += count
                print(f"Total {split} images: {total_images}")
        
        return True
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return False

def main():
    print("=== FER2013 Dataset Setup ===")
    print("This script will help you download and prepare the FER2013 dataset.")
    print()
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Please install kaggle first: pip install kaggle")
        return
    
    # Download dataset
    if download_fer2013_dataset():
        prepare_dataset()
        print("\n=== NEXT STEPS ===")
        print("1. Dataset is ready for training!")
        print("2. Proceed to model training step")
        print("3. Run: python model_training.py")
    else:
        print("\n=== TROUBLESHOOTING ===")
        print("If you have issues with Kaggle API:")
        print("Alternative: Manual download from https://www.kaggle.com/datasets/msambare/fer2013")
        print("Extract to 'data/' folder in your project directory")

if __name__ == "__main__":
    main()