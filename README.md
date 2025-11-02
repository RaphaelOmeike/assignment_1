# 🎭 Face Emotion Detection Web Application

A machine learning web application that detects emotions from facial expressions using deep learning. Built with TensorFlow, Flask, and SQLite.

## 📁 Project Structure

```
FACE_DETECTION/
│
├── app.py                   # Flask web application
├── model_training.py        # CNN model training script
├── init_database.py         # Database initialization
├── download_dataset.py      # Dataset download helper
├── requirements.txt         # Python dependencies
├── database.db             # SQLite database (created after init)
├── face_emotionModel.h5    # Trained model (created after training)
├── link_web_app.txt        # Deployment link information
│
└── templates/
    ├── index.html          # Main upload form
    ├── result.html         # Emotion detection results
    └── users.html          # View all submissions
```

## 🚀 Getting Started

### Step 1: Install Dependencies
```bash
cd FACE_DETECTION
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
- Download FER2013 dataset and place in `data/fer2013/` directory
- Or run: `python download_dataset.py` (requires Kaggle API setup)

### Step 3: Train the Model
```bash
python model_training.py
```
This will:
- Load and preprocess the FER2013 dataset
- Train a CNN model for emotion recognition
- Save the trained model as `face_emotionModel.h5`
- Generate training visualizations

### Step 4: Initialize Database
```bash
python init_database.py
```
This creates the SQLite database with required tables.

### Step 5: Run the Web Application
```bash
python app.py
```
Visit `http://localhost:5000` to use the application.

## 🧠 Model Architecture

The CNN model includes:
- **Input Layer:** 48x48 grayscale images
- **4 Convolutional Blocks:** Feature extraction with increasing complexity
- **Batch Normalization:** Stable training
- **Dropout Layers:** Prevent overfitting
- **Dense Layers:** Final classification
- **Output:** 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)

## 🌐 Web Application Features

### Main Features
- **Photo Upload:** Users can upload images for emotion detection
- **User Information:** Collect name, email, and age
- **Emotion Detection:** AI-powered emotion recognition
- **Confidence Scoring:** Shows prediction confidence
- **Data Storage:** All submissions saved to database
- **Results Visualization:** Interactive emotion breakdown

### Pages
1. **Home Page (`/`):** Upload form with user information
2. **Results Page:** Shows detected emotion with confidence and response
3. **Users Page (`/users`):** View all submissions with filtering

### API Endpoint
- **POST `/api/predict`:** JSON API for emotion prediction

## 🎯 Emotions Detected

| Emotion | Emoji | Response Example |
|---------|-------|------------------|
| Happy | 😊 | "You're glowing with happiness! What's making you smile today?" |
| Sad | 😢 | "You appear sad. It's okay to feel this way sometimes. Things will get better!" |
| Angry | 😠 | "You look angry. What's bothering you? Take a deep breath!" |
| Surprise | 😲 | "You look surprised! Did something unexpected just happen?" |
| Fear | 😨 | "You look worried or scared. Remember, you're brave and can handle this!" |
| Disgust | 🤢 | "You seem disgusted. Is something not quite right?" |
| Neutral | 😐 | "You have a calm, neutral expression. Sometimes that's the perfect mood!" |

## 💾 Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    age INTEGER NOT NULL,
    predicted_emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    image_filename TEXT NOT NULL,
    image_data BLOB,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🛠 Technology Stack

- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript (no external frameworks)
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## 📊 Model Performance

Expected performance metrics:
- **Accuracy:** 60-70% (typical for emotion recognition)
- **Training Time:** 30-60 minutes
- **Model Size:** ~50MB
- **Inference Time:** <1 second per image

## 🌐 Deployment

### For Render Deployment:
1. Push code to GitHub repository
2. Connect GitHub repo to Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python app.py`
5. Update `link_web_app.txt` with deployment URL

### Environment Variables:
- `PORT`: Set by Render automatically
- `FLASK_ENV`: production

## 🔧 Troubleshooting

### Common Issues:

1. **Model not found error:**
   - Ensure `python model_training.py` completed successfully
   - Check that `face_emotionModel.h5` exists

2. **Database errors:**
   - Run `python init_database.py` to recreate database
   - Check file permissions

3. **Import errors:**
   - Install all requirements: `pip install -r requirements.txt`
   - Use virtual environment if needed

4. **No face detected:**
   - Ensure uploaded image contains a clear face
   - Try images with better lighting and front-facing poses

## 📈 Future Improvements

- [ ] Real-time emotion detection via webcam
- [ ] Multi-face detection and emotion analysis
- [ ] Improved model accuracy with data augmentation
- [ ] User authentication and personal history
- [ ] Emotion trend analysis over time
- [ ] Mobile app version

## 📝 Notes

- Model training requires significant computational resources
- Dataset not included due to size (download separately)
- All user data stored locally in SQLite database
- Images stored as binary data in database

## 🤝 Contributing

This project was created as an educational example. Feel free to:
- Improve the model architecture
- Add new features
- Enhance the user interface
- Optimize performance

---

**Made with ❤️ using AI and Machine Learning**