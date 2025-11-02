"""
Face Emotion Detection - Flask Web Application
This app allows users to upload photos and detect emotions.
"""

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from PIL import Image
import base64
from io import BytesIO

# ============================================
# FLASK APP INITIALIZATION
# ============================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = 'face_emotionModel.h5'
DATABASE_PATH = 'database.db'
IMG_SIZE = 48

# Emotion labels (must match training order)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion responses (personalized messages)
EMOTION_RESPONSES = {
    'angry': "You look angry. Is everything okay? Take a deep breath! 😤",
    'disgust': "You seem disgusted. Something bothering you? 🤢",
    'fear': "You look fearful. Don't worry, everything will be alright! 😨",
    'happy': "You're happy! That's wonderful! Keep smiling! 😊",
    'sad': "You look sad. Why are you sad? Cheer up, things will get better! 😢",
    'surprise': "You look surprised! What's the big news? 😲",
    'neutral': "You have a neutral expression. Feeling calm today? 😐"
}

# ============================================
# LOAD MODEL
# ============================================
print("Loading emotion detection model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# ============================================
# DATABASE SETUP
# ============================================
def init_database():
    """Initialize SQLite database with users table"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create table for storing user data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized!")

# Initialize database on startup
init_database()

# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image):
    """
    Preprocess uploaded image for model prediction
    Args:
        image: PIL Image object
    Returns:
        Preprocessed numpy array ready for prediction
    """
    try:
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to model input size (48x48)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to 0-1
        image_array = image_array / 255.0
        
        # Reshape to match model input shape: (1, 48, 48, 1)
        image_array = image_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return image_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# ============================================
# EMOTION PREDICTION
# ============================================
def predict_emotion(image_array):
    """
    Predict emotion from preprocessed image
    Args:
        image_array: Preprocessed numpy array
    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    try:
        if model is None:
            return None, 0.0
        
        # Get prediction probabilities
        predictions = model.predict(image_array, verbose=0)
        
        # Get the emotion with highest probability
        emotion_index = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_index])
        emotion = EMOTIONS[emotion_index]
        
        return emotion, confidence
    
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, 0.0

# ============================================
# DATABASE OPERATIONS
# ============================================
def save_to_database(name, email, age, gender, emotion, confidence, image_base64):
    """
    Save user data and emotion result to database
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (name, email, age, gender, emotion, confidence, image_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, age, gender, emotion, confidence, image_base64))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        print(f"✓ User data saved to database (ID: {user_id})")
        return True
    
    except Exception as e:
        print(f"✗ Error saving to database: {e}")
        return False

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and emotion prediction
    """
    try:
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        
        # Validate required fields
        if not name or not email:
            return jsonify({
                'success': False,
                'error': 'Name and email are required!'
            })
        
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image uploaded!'
            })
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected!'
            })
        
        # Read and process image
        image = Image.open(file.stream)
        
        # Convert image to base64 for storage
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Preprocess image for prediction
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': 'Error processing image!'
            })
        
        # Predict emotion
        emotion, confidence = predict_emotion(processed_image)
        
        if emotion is None:
            return jsonify({
                'success': False,
                'error': 'Error predicting emotion!'
            })
        
        # Save to database
        age_int = int(age) if age.isdigit() else None
        db_saved = save_to_database(name, email, age_int, gender, emotion, confidence, image_base64)
        
        if not db_saved:
            print("Warning: Failed to save to database, but continuing...")
        
        # Get personalized response
        response_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
        
        # Return result
        return jsonify({
            'success': True,
            'emotion': emotion.capitalize(),
            'confidence': round(confidence * 100, 2),
            'message': response_message,
            'name': name
        })
    
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })

@app.route('/stats')
def stats():
    """
    View database statistics (optional admin page)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM users 
            GROUP BY emotion 
            ORDER BY count DESC
        ''')
        emotion_counts = cursor.fetchall()
        
        conn.close()
        
        stats_html = f"""
        <html>
        <head>
            <title>Statistics</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
                h1 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>📊 Database Statistics</h1>
            <p><strong>Total Users:</strong> {total_users}</p>
            <h2>Emotion Distribution</h2>
            <table>
                <tr>
                    <th>Emotion</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for emotion, count in emotion_counts:
            percentage = (count / total_users * 100) if total_users > 0 else 0
            stats_html += f"""
                <tr>
                    <td>{emotion.capitalize()}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        stats_html += """
            </table>
            <br>
            <a href="/">← Back to Home</a>
        </body>
        </html>
        """
        
        return stats_html
    
    except Exception as e:
        return f"<h1>Error loading statistics: {e}</h1>"

# ============================================
# RUN APP
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("FACE EMOTION DETECTION - WEB APP")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)