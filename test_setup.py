"""
Test script to verify the application setup before training the model.
This script checks if all components are working correctly.
"""

import os
import sys

def check_files():
    """Check if all required files exist."""
    print("=== Checking Project Files ===")
    
    required_files = [
        'app.py',
        'model_training.py',
        'init_database.py',
        'requirements.txt',
        'templates/index.html',
        'templates/result.html',
        'templates/users.html'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def check_python_syntax():
    """Check if Python files have correct syntax."""
    print("\n=== Checking Python Syntax ===")
    
    python_files = ['app.py', 'model_training.py', 'init_database.py']
    syntax_errors = []
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                content = f.read()
            compile(content, file, 'exec')
            print(f"✅ {file} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {file} - syntax error: {e}")
            syntax_errors.append(file)
        except Exception as e:
            print(f"⚠️  {file} - could not check: {e}")
    
    if syntax_errors:
        print(f"\n⚠️  Files with syntax errors: {syntax_errors}")
        return False
    else:
        print("\n✅ All Python files have correct syntax!")
        return True

def check_imports():
    """Check which required packages are installed."""
    print("\n=== Checking Package Availability ===")
    
    required_packages = {
        'flask': 'Flask web framework',
        'tensorflow': 'TensorFlow for machine learning',
        'numpy': 'NumPy for numerical operations',
        'opencv-python': 'OpenCV for image processing',
        'pandas': 'Pandas for data manipulation',
        'matplotlib': 'Matplotlib for plotting',
        'scikit-learn': 'Scikit-learn for metrics',
        'seaborn': 'Seaborn for visualization',
        'pillow': 'PIL/Pillow for image handling',
        'werkzeug': 'Werkzeug (Flask dependency)'
    }
    
    installed = []
    missing = []
    
    for package, description in required_packages.items():
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ {package} ({description})")
                installed.append(package)
            elif package == 'pillow':
                from PIL import Image
                print(f"✅ {package} ({description})")
                installed.append(package)
            else:
                __import__(package)
                print(f"✅ {package} ({description})")
                installed.append(package)
        except ImportError:
            print(f"❌ {package} ({description})")
            missing.append(package)
    
    print(f"\n📊 Summary: {len(installed)}/{len(required_packages)} packages available")
    
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def test_database():
    """Test database initialization."""
    print("\n=== Testing Database ===")
    
    try:
        # Import the database manager
        sys.path.append('.')
        from init_database import init_database
        
        # Test database creation
        test_db = 'test_database.db'
        if os.path.exists(test_db):
            os.remove(test_db)
        
        success = init_database(test_db)
        
        if success and os.path.exists(test_db):
            print("✅ Database initialization works!")
            os.remove(test_db)  # Clean up
            return True
        else:
            print("❌ Database initialization failed!")
            return False
            
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_flask_import():
    """Test if Flask app can be imported without errors."""
    print("\n=== Testing Flask App Import ===")
    
    try:
        # Try to import the Flask app components
        import sqlite3
        from datetime import datetime
        
        print("✅ Basic imports successful")
        
        # Try Flask import separately
        try:
            from flask import Flask
            print("✅ Flask import successful")
            flask_available = True
        except ImportError:
            print("❌ Flask not available")
            flask_available = False
        
        return flask_available
        
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 FACE EMOTION DETECTION - PROJECT VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("File Structure", check_files),
        ("Python Syntax", check_python_syntax),
        ("Package Availability", check_imports),
        ("Database Functionality", test_database),
        ("Flask App Import", test_flask_import)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Ready to:")
        print("1. Install packages: pip install -r requirements.txt")
        print("2. Train model: python model_training.py")
        print("3. Initialize database: python init_database.py")
        print("4. Run app: python app.py")
    else:
        print("\n⚠️  Some tests failed. Please:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Fix any syntax errors")
        print("3. Re-run this test script")

if __name__ == "__main__":
    main()