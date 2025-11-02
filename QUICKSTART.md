# 🚀 Quick Start Guide

This guide will get your Face Emotion Detection app running quickly!

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- At least 2GB free disk space

## Step 1: Install Packages

### Option A: Automatic Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
```

### Option B: Manual Setup
```bash
pip install -r requirements.txt
```

## Step 2: Test Installation
```bash
python test_setup.py
```
This will verify that everything is installed correctly.

## Step 3: Get Dataset (Choose One Option)

### Option A: Download FER2013 (Recommended)
1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download the dataset
3. Extract to `data/fer2013/` folder in your project

### Option B: Use Kaggle API
```bash
# First setup Kaggle API credentials
python download_dataset.py
```

### Expected Dataset Structure:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Step 4: Train the Model
```bash
python model_training.py
```
⏰ **This takes 30-60 minutes depending on your hardware**

Expected output:
- `face_emotionModel.h5` (trained model)
- `training_history.png` (training graphs)
- `confusion_matrix.png` (performance matrix)

## Step 5: Initialize Database
```bash
python init_database.py
```
This creates `database.db` for storing user submissions.

## Step 6: Run the Web App
```bash
python app.py
```
🌐 Visit: http://localhost:5000

## Troubleshooting

### ❌ "Module not found" errors
```bash
pip install -r requirements.txt
```

### ❌ "Model not found" error
Make sure you completed Step 4 (train the model)

### ❌ "Dataset not found" error
Make sure you completed Step 3 (download dataset)

### ❌ Permission errors on macOS/Linux
```bash
chmod +x setup.sh
```

## Quick Test Without Training

If you want to test the web interface before training:

1. Install packages: `pip install -r requirements.txt`
2. Initialize database: `python init_database.py`
3. Run app: `python app.py`
4. Visit: http://localhost:5000

The app will show an error message that the model needs to be trained first, but you can see the interface!

## What to Expect

### After Training:
- Model accuracy: ~60-70%
- Model file size: ~50MB
- Inference time: <1 second per image

### Web Interface:
- Upload form for photos and user info
- Emotion detection results with confidence
- Database of all submissions
- Interactive emotion breakdown charts

## Next Steps After Setup

1. **Test with Photos**: Upload clear face photos for best results
2. **View Results**: Check the emotion predictions and confidence scores
3. **Explore Data**: Use the `/users` page to see all submissions
4. **Deploy**: Follow deployment instructions for Render hosting

## Need Help?

1. Run `python test_setup.py` to diagnose issues
2. Check that all files are present
3. Ensure Python 3.8+ is installed
4. Make sure you have enough disk space

---
Happy emotion detecting! 🎭