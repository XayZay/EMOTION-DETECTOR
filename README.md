# Face Emotion Detection Web App

A machine learning web application that detects emotions from facial images using a Convolutional Neural Network (CNN).

## Features
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Web interface for image upload
- Stores user data in SQLite database
- Personalized emotion responses

## Tech Stack
- **Backend:** Flask, TensorFlow/Keras
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **Deployment:** Render

## Local Setup

1. Clone the repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the app:
```bash
   python app.py
```
4. Open browser: `http://127.0.0.1:5000`

## Model Training

To train the model:
1. Download FER2013 dataset
2. Place in `fer2013/` folder
3. Run:
```bash
   python model_training.py
```

## Deployment

Deployed on Render: [Your deployment link will go here]

---

## 📂 Final Project Structure

Before deployment, your folder should look like this:
```
FACE_DETECTION/
│
├── templates/
│   └── index.html
│
├── app.py
├── model_training.py
├── requirements.txt
├── Procfile                  
├── runtime.txt               
├── README.md                 
├── .gitignore
├── .gitattributes
├── link_web_app.txt
│
├── face_emotionModel.h5      # Your trained model 
├── database.db               # (will be created/reset on deployment)
└── training_history.png      # (optional)