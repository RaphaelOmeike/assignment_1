"""
Face Emotion Recognition Model Training
This script creates and trains a CNN model to detect emotions from facial expressions.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class EmotionModelTrainer:
    def __init__(self, data_path="archive"):
        """
        Initialize the emotion model trainer.
        
        Args:
            data_path (str): Path to the FER2013 dataset
        """
        self.data_path = data_path
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.num_classes = len(self.emotion_labels)
        self.img_height = 48
        self.img_width = 48
        self.batch_size = 32
        self.model = None
        self.history = None
        
    def load_data(self):
        """
        Load and preprocess the training and validation data.
        """
        print("Loading dataset...")
        
        # Create data generators with augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to [0,1]
            rotation_range=10,  # Randomly rotate images
            width_shift_range=0.1,  # Randomly shift images horizontally
            height_shift_range=0.1,  # Randomly shift images vertically
            horizontal_flip=True,  # Randomly flip images horizontally
            fill_mode='nearest',
            validation_split=0.2  # Use 20% of training data for validation
        )
        
        # Create data generator for test data (no augmentation)
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            subset='validation',
            shuffle=True
        )
        
        # Load test data
        self.test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_path, 'test'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Class labels: {self.train_generator.class_indices}")
        
    def create_model(self):
        """
        Create the CNN model architecture.
        """
        print("Creating CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(48, 48, 1)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train_model(self, epochs=50):
        """
        Train the model with callbacks for best performance.
        
        Args:
            epochs (int): Number of training epochs
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Callbacks for better training
        callbacks = [
            # Save the best model
            keras.callbacks.ModelCheckpoint(
                'face_emotionModel.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Reduce learning rate when stuck
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            # Stop early if no improvement
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """
        Evaluate the model performance on test data.
        """
        print("Evaluating model on test data...")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions for detailed analysis
        print("Generating predictions...")
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.emotion_labels))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy
    
    def plot_training_history(self):
        """
        Plot training history graphs.
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='face_emotionModel.h5'):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            print("No model to save. Please train the model first.")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save model architecture as JSON
        model_json = self.model.to_json()
        with open("model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        print("Model architecture saved to model_architecture.json")

def test_single_prediction():
    """
    Test function to verify the model can make predictions.
    """
    print("\n=== Testing Single Prediction ===")
    
    # Load the saved model
    try:
        model = keras.models.load_model('face_emotionModel.h5')
        print("Model loaded successfully!")
        
        # Create a dummy image for testing
        dummy_image = np.random.rand(1, 48, 48, 1)
        prediction = model.predict(dummy_image)
        predicted_emotion = np.argmax(prediction)
        
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        print(f"Test prediction: {emotion_labels[predicted_emotion]}")
        print("Model is ready for use!")
        
    except Exception as e:
        print(f"Error testing model: {e}")

def main():
    """
    Main function to train the emotion recognition model.
    """
    print("=== Face Emotion Recognition Model Training ===")
    print("This will create and train a CNN model to detect emotions from facial expressions.\n")
    
    # Initialize trainer
    trainer = EmotionModelTrainer()
    
    # Check if data path exists
    if not os.path.exists(trainer.data_path):
        print(f"Error: Dataset not found at {trainer.data_path}")
        print("Please ensure you have the FER2013 dataset downloaded and extracted.")
        print("Expected structure:")
        print("archive/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   ├── fear/")
        print("│   ├── happy/")
        print("│   ├── neutral/")
        print("│   ├── sad/")
        print("│   └── surprise/")
        print("└── test/")
        print("    ├── angry/")
        print("    ├── disgust/")
        print("    ├── fear/")
        print("    ├── happy/")
        print("    ├── neutral/")
        print("    ├── sad/")
        print("    └── surprise/")
        return
    
    try:
        # Load data
        trainer.load_data()
        
        # Create model
        trainer.create_model()
        
        # Train model
        print("\nStarting training... This may take 10-15 minutes depending on your hardware.")
        print("The model will automatically save the best version during training.")
        trainer.train_model(epochs=10)
        
        # Evaluate model
        accuracy = trainer.evaluate_model()
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save final model
        trainer.save_model()
        
        # Test the model
        test_single_prediction()
        
        print(f"\n=== Training Complete! ===")
        print(f"Final test accuracy: {accuracy:.4f}")
        print("Model saved as: face_emotionModel.h5")
        print("Ready to proceed to Flask web app development!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your dataset structure and try again.")

if __name__ == "__main__":
    main()