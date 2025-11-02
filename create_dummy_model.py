"""
Quick Model Simulation for Testing
This creates a simple dummy model for testing the web app functionality
"""
import numpy as np

def create_dummy_model():
    """Create a dummy model for testing purposes"""
    print("Creating dummy model for testing...")
    
    # Import TensorFlow
    try:
        import tensorflow as tf
        
        # Create a simple sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(48, 48, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize with random weights (for testing)
        # The model won't be accurate but will function
        dummy_input = np.random.random((1, 48, 48, 1))
        _ = model.predict(dummy_input, verbose=0)
        
        # Save the dummy model
        model.save('face_emotionModel.h5')
        print("✅ Dummy model created and saved as 'face_emotionModel.h5'")
        print("⚠️  This is for testing only - predictions will be random!")
        
        return True
        
    except Exception as e:
        print(f"Error creating dummy model: {e}")
        return False

if __name__ == "__main__":
    create_dummy_model()