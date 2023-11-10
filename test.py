from keras.models import load_model
import librosa
import numpy as np
import os
from train_cfg import hop_length, n_fft, n_mfcc, duration, sample_rate, audio_dir

best_model = load_model('yugi-best-trained-model-custom-checkpoint.tf')

audio_files = os.listdir(audio_dir)

labels = [os.path.splitext(file)[0] for file in audio_files]

user_audio_file = input('Audio Path: ')

def predict_audio_label(model, audio_file):
    audio_data = preprocess_audio(audio_file)
    predictions = model.predict(np.expand_dims(audio_data, axis=0))
    predicted_label = labels[np.argmax(predictions)]
    return predicted_label

def predict_top_3_labels(model, audio_file):
    audio_data = preprocess_audio(audio_file)
    predictions = model.predict(np.expand_dims(audio_data, axis=0))
    
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    
    top_3_labels = [labels[i] for i in top_3_indices]
    top_3_probabilities = [predictions[0][i] * 100 for i in top_3_indices]
    
    return top_3_labels, top_3_probabilities

def preprocess_audio(audio_file):

    audio, sr = librosa.load(audio_file, mono=True, duration=duration)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    if mfccs.shape[1] < 1292:
        mfccs = np.pad(mfccs, ((0, 0), (0, 1292 - mfccs.shape[1])), mode='constant')

    return mfccs

predicted_label = predict_audio_label(best_model, user_audio_file)
print(f"I think, it's: {predicted_label}")

predicted_top_3_labels, predicted_top_3_probabilities = predict_top_3_labels(best_model, user_audio_file)
print("Top 3 predicted labels with probabilities:")
for i, (label, probability) in enumerate(zip(predicted_top_3_labels, predicted_top_3_probabilities), start=1):
    print(f"Top {i}: {label} (Probability: {probability:.2f}%)")