import gradio as gr
import librosa
import numpy as np
import joblib
from scipy import stats

# Load model
try:
    model = joblib.load('lstm_student_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

import librosa
import numpy as np
from scipy import stats

def extract_features(y, sr):
    try:
        print(f"🔍 Debug: Received y -> {type(y)}, sr -> {type(sr)}")

        if not isinstance(y, np.ndarray) or not isinstance(sr, (int, np.integer)):
            raise ValueError(f"Invalid audio data! y={type(y)}, sr={type(sr)}")

        print(f"✅ Original Audio shape: {y.shape}, Sample Rate: {sr}")

        # Ensure y is a 1D array
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Normalize
        y = librosa.util.normalize(y)

        print(f"✅ Processed Audio shape: {y.shape}, dtype: {y.dtype}")

        hop_length = 512
        n_fft = 2048
        # Existing features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
        mel_spec = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        skew = stats.skew(y)
        kurtosis = stats.kurtosis(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        beats = len(librosa.beat.beat_track(y=y, sr=sr)[1])
    
        features = np.hstack((mfccs, spectral_centroid, spectral_bandwidth, spectral_rolloff,
                              zero_crossing_rate, chroma_stft, spectral_contrast,
                              rmse, mel_spec, skew, kurtosis, tempo, beats))

        # Reshape for LSTM input (add time steps dimension)
        features = features.reshape(1, 1, -1)

        print(f"✅ Feature shape: {features.shape}")  # Debug log
        return features
    except Exception as e:

        return features
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

def classify_audio(audio):
    try:
        print(f"🔍 Debug: Received audio type -> {type(audio)}")

        if isinstance(audio, tuple) and len(audio) == 2:
            y, sr = audio
            if isinstance(y, int) and isinstance(sr, np.ndarray):  
                print("❌ Detected swapped values. Fixing...")
                y, sr = sr, y

        # Convert audio to float32
        y = y.astype(np.float32)
        
        # Ensure y is in the correct shape (should be 1D)
        if len(y.shape) == 2:
            y = y.mean(axis=1)  # Convert stereo to mono by averaging channels

        features = extract_features(y, sr)
        if features is None:
            return "Error extracting features!"
        
        prediction = model.predict(features)
        # Handle the prediction output properly
        prediction = prediction.squeeze()  # Remove any extra dimensions
        if prediction.size > 1:
            # If you have multiple output classes, use argmax
            predicted_class = np.argmax(prediction)
            return "Healthy" if predicted_class == 1 else "Unhealthy"
        else:
            # If you have a single output (binary classification)
            return "Healthy" if prediction > 0.5 else "Unhealthy"
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error during prediction: {str(e)}"


iface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="numpy"),
    outputs="text",
    title="Poultry Disease Detection from Audio",
    description="Upload an audio file of poultry vocalization to detect potential diseases."
)

iface.launch()