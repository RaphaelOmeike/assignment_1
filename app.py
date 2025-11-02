"""
Flask Web Application for Face Emotion Detection
This app allows users to upload images and get emotion predictions.
"""

import os
import sqlite3
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels (must match the order from training)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion responses for user interaction
EMOTION_RESPONSES = {
    'angry': "You look angry. What's bothering you? Take a deep breath!",
    'disgust': "You seem disgusted. Is something not quite right?",
    'fear': "You look worried or scared. Remember, you're brave and can handle this!",
    'happy': "You're glowing with happiness! What's making you smile today?",
    'neutral': "You have a calm, neutral expression. Sometimes that's the perfect mood!",
    'sad': "You appear sad. It's okay to feel this way sometimes. Things will get better!",
    'surprise': "You look surprised! Did something unexpected just happen?"
}

class EmotionDetector:
    def __init__(self, model_path='face_emotionModel.h5'):
        """Initialize the emotion detector with the trained model."""
        self.model = None
        self.model_path = model_path
        self.model_available = False
        self.load_model()
        
    def load_model(self):
        """Load the trained emotion detection model."""
        try:
            if os.path.exists(self.model_path):
                # Only try to import tensorflow if model file exists
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                self.model_available = True
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Warning: Model file {self.model_path} not found!")
                print("Please train the model first by running: python model_training.py")
                self.model_available = False
        except ImportError:
            print("TensorFlow not installed. Please install requirements: pip install -r requirements.txt")
            self.model_available = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_available = False
    
    def preprocess_image(self, image_path):
        """
        Preprocess the uploaded image for emotion detection.
        
        Args:
            image_path (str): Path to the uploaded image
            
        Returns:
            np.array: Preprocessed image ready for prediction
        """
        try:
            # Import libraries only when needed
            import cv2
            import numpy as np
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face using OpenCV's face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                # If no face detected, use the entire image
                print("No face detected, using entire image")
                face_img = gray
            else:
                # Use the first (largest) detected face
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 (model input size)
            face_img = cv2.resize(face_img, (48, 48))
            
            # Normalize pixel values to [0, 1]
            face_img = face_img.astype('float32') / 255.0
            
            # Reshape for model input: (1, 48, 48, 1)
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)
            
            return face_img
            
        except ImportError as e:
            print(f"Missing required library: {e}")
            print("Please install requirements: pip install -r requirements.txt")
            return None
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_emotion(self, image_path):
        """
        Predict emotion from an uploaded image.
        
        Args:
            image_path (str): Path to the uploaded image
            
        Returns:
            dict: Prediction results with emotion and confidence
        """
        if not self.model_available or self.model is None:
            return {
                'success': False,
                'error': 'Model not available. Please train the model first by running: python model_training.py',
                'emotion': 'neutral',
                'confidence': 0.0,
                'response': 'Unable to detect emotion. Please train the model first.'
            }
        
        try:
            # Import numpy only when needed
            import numpy as np
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return {
                    'success': False,
                    'error': 'Could not process image',
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'response': 'Unable to process your image.'
                }
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Get emotion label
            emotion = EMOTION_LABELS[predicted_class]
            response = EMOTION_RESPONSES[emotion]
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'response': response,
                'all_predictions': {
                    EMOTION_LABELS[i]: float(predictions[0][i]) 
                    for i in range(len(EMOTION_LABELS))
                }
            }
            
        except ImportError as e:
            print(f"Missing required library: {e}")
            return {
                'success': False,
                'error': f'Missing required library: {e}',
                'emotion': 'neutral',
                'confidence': 0.0,
                'response': 'Please install required packages: pip install -r requirements.txt'
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e),
                'emotion': 'neutral',
                'confidence': 0.0,
                'response': 'Something went wrong during emotion detection.'
            }

class DatabaseManager:
    def __init__(self, db_path='database.db'):
        """Initialize the database manager."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create the database and tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    age INTEGER,
                    predicted_emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_filename TEXT NOT NULL,
                    image_data BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def save_user_data(self, name, email, age, emotion, confidence, image_filename, image_path):
        """
        Save user data and image to the database.
        
        Args:
            name (str): User's name
            email (str): User's email
            age (int): User's age
            emotion (str): Predicted emotion
            confidence (float): Prediction confidence
            image_filename (str): Original filename
            image_path (str): Path to saved image
        
        Returns:
            bool: Success status
        """
        try:
            # Read image as binary data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (name, email, age, predicted_emotion, confidence, 
                                 image_filename, image_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, email, age, emotion, confidence, image_filename, image_data))
            
            conn.commit()
            conn.close()
            
            print(f"User data saved: {name} - {emotion}")
            return True
            
        except Exception as e:
            print(f"Error saving user data: {e}")
            return False
    
    def get_all_users(self):
        """Get all users from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, email, age, predicted_emotion, confidence, 
                       image_filename, timestamp
                FROM users ORDER BY timestamp DESC
            ''')
            
            users = cursor.fetchall()
            conn.close()
            
            return users
            
        except Exception as e:
            print(f"Error fetching users: {e}")
            return []

# Initialize components
detector = EmotionDetector()
db_manager = DatabaseManager()

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and emotion detection."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded.')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(url_for('index'))
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            flash('Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP).')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        # Add timestamp to filename to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect emotion
        result = detector.predict_emotion(filepath)
        
        if result['success']:
            # Save to database with default values for removed fields
            db_success = db_manager.save_user_data(
                "Anonymous", "anonymous@email.com", 0, 
                result['emotion'], result['confidence'],
                filename, filepath
            )
            
            if db_success:
                flash(f"Success! {result['response']}")
                return render_template('result.html', 
                                     name="Anonymous",
                                     emotion=result['emotion'],
                                     confidence=result['confidence'],
                                     response=result['response'],
                                     all_predictions=result.get('all_predictions', {}))
            else:
                flash('Emotion detected but failed to save data.')
                return redirect(url_for('index'))
        else:
            flash(f"Error: {result.get('error', 'Unknown error occurred')}")
            return redirect(url_for('index'))
            
    except Exception as e:
        print(f"Error in upload_file: {e}")
        flash('An unexpected error occurred. Please try again.')
        return redirect(url_for('index'))

@app.route('/users')
def view_users():
    """View all users and their data."""
    users = db_manager.get_all_users()
    return render_template('users.html', users=users)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for emotion prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
        file.save(temp_path)
        
        # Predict emotion
        result = detector.predict_emotion(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=== Face Emotion Detection Web App ===")
    print("Starting Flask server...")
    print("Visit http://localhost:5000 to use the application")
    print("Press Ctrl+C to stop the server")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)