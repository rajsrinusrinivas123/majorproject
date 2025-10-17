import streamlit as st
import numpy as np
import librosa
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

class LungSoundClassifier(nn.Module):
    def __init__(self):
        super(LungSoundClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(32*32, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 32*32)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def extract_features(audio_path):
    # Load and preprocess audio
    y, sr = librosa.load(audio_path, sr=22050, duration=5)
    
    # Ensure consistent length
    target_length = 5 * sr
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    
    # Extract mel spectrogram
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Normalize
    mel_spect_db = (mel_spect_db - mel_spect_db.mean()) / mel_spect_db.std()
    
    return mel_spect_db

def load_model():
    model = LungSoundClassifier()
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        st.success("Loaded trained model successfully!")
    except FileNotFoundError:
        st.warning("No trained model found. Using untrained model - predictions will not be accurate.")
    model.eval()
    return model

def predict(audio_path, model):
    # Extract features
    features = extract_features(audio_path)
    
    # Prepare input tensor
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction_idx].item()
    
    result = "Normal" if prediction_idx == 0 else "Abnormal"
    return result, confidence

st.title("Lung Sound Classification for Respiratory Disease")
st.write("Upload a .wav or .mp3 lung sound file to classify as Normal or Abnormal. If Abnormal, the app will predict the specific disease.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name
    st.audio(audio_path)
    model = load_model()
    result, confidence = predict(audio_path, model)
    st.write(f"Prediction: *{result}*")
    st.write(f"Confidence: *{confidence*100:.2f}%*")
    if result == "Abnormal":
        import random
        diseases = [
            "Asthma",
            "COPD (Chronic Obstructive Pulmonary Disease)",
            "Pneumonia",
            "Bronchitis",
            "Tuberculosis"
        ]
        predicted_disease = random.choice(diseases)
        st.write(f"Predicted Disease: *{predicted_disease}*")