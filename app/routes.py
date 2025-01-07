# File: app/routes.py
from flask import request, render_template, redirect, url_for, jsonify
from app import app
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import joblib
import smtplib
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import os

# Load the pre-trained model and LabelEncoder
model = load_model("./model/my_model.h5")
labelencoder = joblib.load("./model/label_encoder.pkl")

# Email configuration
EMAIL_ADDRESS = "prabhat.bhasme21@vit.edu"  # Replace with your email
EMAIL_PASSWORD = "gjmv jdyg txlw ionx"  
RECIPIENTS = [
    "prabhatbhasme@gmail.com",
    "sachin.bhargav21@vit.edu",
]
# FireBase
cred = credentials.Certificate("firebase_key.json")

if not firebase_admin._apps:  # Check if Firebase is already initialized
    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": "https://audio-classification-c3e30-default-rtdb.firebaseio.com"  # Replace with your Firebase database URL
        },
    )


def send_email(predicted_class):
    SUBJECT = f"Alert: {predicted_class} Detected"
    BODY = f"This is an automated alert for the detected class: {predicted_class}. Please take necessary actions."
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        for recipient in RECIPIENTS:
            server.sendmail(EMAIL_ADDRESS, recipient, f"Subject: {SUBJECT}\n\n{BODY}")
    print("Email sent successfully!")


def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/project")
def project():
    return render_template("project.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("project"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("project"))

    # Save the uploaded file locally
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Extract features and make prediction
    mfccs_scaled_features = features_extractor(file_path).reshape(1, -1)
    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)[0]

    # Clean up the uploaded file locally
    os.remove(file_path)

    # Store metadata in Firebase Realtime Database
    timestamp = datetime.now().isoformat()  # Current time in ISO format
    db.reference("audio_records").push(
        {
            "audio_filename": file.filename,
            "classified_audio": prediction_class,
            "timestamp": timestamp,
        }
    )
    print("Data stored in Firebase Realtime Database successfully!")

    # Send email notification
    send_email(predicted_class=prediction_class)

    # Display the prediction result
    return render_template("project.html", prediction=prediction_class)
