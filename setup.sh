#!/bin/bash

# Face Emotion Detection - Setup Script
# This script installs all required packages and sets up the project

echo "🎭 Face Emotion Detection - Setup Script"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip3 install --upgrade pip

# Install requirements
echo ""
echo "📦 Installing required packages..."
echo "This may take several minutes depending on your internet connection..."

pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Package installation failed. Please check the error messages above."
    exit 1
fi

# Test the installation
echo ""
echo "🧪 Testing installation..."
python3 test_setup.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download FER2013 dataset (or run: python3 download_dataset.py)"
echo "2. Train the model: python3 model_training.py"
echo "3. Initialize database: python3 init_database.py"
echo "4. Run the web app: python3 app.py"
echo ""
echo "Visit http://localhost:5000 to use the application!"