#!/usr/bin/env python3.11

"""
Face Emotion Recognition - Quick Start Script
This script ensures we use the correct Python version (3.11) for all operations.
"""

import sys
import subprocess
import os

def run_with_python311(script_name):
    """Run a Python script with python3.11"""
    try:
        result = subprocess.run([
            'python3.11', script_name
        ], cwd=os.getcwd(), capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Main menu for the project"""
    print("🎭 FACE EMOTION DETECTION - Quick Launcher")
    print("==========================================")
    print("Using Python 3.11 with TensorFlow 2.20.0")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Train the model (python3.11 model_training.py)")
        print("2. Initialize database (python3.11 init_database.py)")
        print("3. Run web app (python3.11 app.py)")
        print("4. Test setup (python3.11 test_setup.py)")
        print("5. Exit")
        print()
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            print("\n🧠 Starting model training...")
            print("This will take 30-60 minutes. Press Ctrl+C to stop.")
            success = run_with_python311('model_training.py')
            if success:
                print("\n✅ Model training completed!")
            else:
                print("\n❌ Model training failed!")
        
        elif choice == '2':
            print("\n💾 Initializing database...")
            success = run_with_python311('init_database.py')
            if success:
                print("\n✅ Database initialized!")
            else:
                print("\n❌ Database initialization failed!")
        
        elif choice == '3':
            print("\n🌐 Starting web application...")
            print("Visit http://localhost:5000 after startup")
            print("Press Ctrl+C to stop the server")
            run_with_python311('app.py')
        
        elif choice == '4':
            print("\n🧪 Running setup tests...")
            run_with_python311('test_setup.py')
        
        elif choice == '5':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()