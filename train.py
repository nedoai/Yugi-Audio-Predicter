import os
import librosa
import numpy as np
import tensorflow as tf

from keras.callbacks import Callback
from keras.optimizers import Adam

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from train_cfg import epochs, batch_size, hop_length, n_fft, n_mfcc, duration, sample_rate, audio_dir, learning_rate

from keras.callbacks import Callback

from model_file import Yugi

class CustomModelCheckpoint(Callback):
    def __init__(self, model, save_path, monitor='val_loss', mode='min'):
        super(CustomModelCheckpoint, self).__init__()
        self.model = model
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_value = None

        if mode == 'min':
            self.best_value = float('inf')
        elif mode == 'max':
            self.best_value = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        
        if self.mode == 'min' and current_value < self.best_value:
            self.best_value = current_value
            self.model.save(self.save_path)
            print(f"Model saved at epoch {epoch} with {self.monitor} of {current_value}")
        elif self.mode == 'max' and current_value > self.best_value:
            self.best_value = current_value
            self.model.save(self.save_path)
            print(f"Model saved at epoch {epoch} with {self.monitor} of {current_value}")


audio_files = os.listdir(audio_dir)
labels = [os.path.splitext(file)[0] for file in audio_files]


"""Audio Augmentation. Set the values as desired here"""

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),

    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
])



def preprocess_audio(audio_file):

    audio, sr = librosa.load(audio_file, mono=True, duration=duration)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    if mfccs.shape[1] < 1292:
        mfccs = np.pad(mfccs, ((0, 0), (0, 1292 - mfccs.shape[1])), mode='constant')

    return mfccs

train_data = []
train_labels = []

for audio_file, label in zip(audio_files, labels):
    audio_path = os.path.join(audio_dir, audio_file)
    audio_data = preprocess_audio(audio_path)

    augmented_samples = augment(samples=audio_data, sample_rate=sample_rate)

    train_data.append(augmented_samples)
    train_labels.append(labels.index(label))

    train_data.append(audio_data)
    train_labels.append(labels.index(label))

max_len = max(len(audio_data[0]) for audio_data in train_data)

train_data = np.array([audio_data[:, :max_len] if audio_data.shape[1] > max_len else np.pad(audio_data, ((0, 0), (0, max_len - audio_data.shape[1])), mode='constant') for audio_data in train_data])


train_labels = np.array(train_labels)

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)

input_shape = train_data.shape[1:]
num_classes = len(labels)

optimizer = Adam(learning_rate=learning_rate)

model = Yugi(input_shape=input_shape, num_classes=num_classes)

model.compile_model(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.get_model_summary()

custom_checkpoint = CustomModelCheckpoint(model, 'yugi-best-trained-model-custom-checkpoint.tf', monitor='loss', mode='min')
model.train(train_data, train_labels, epochs=epochs, batch_size=batch_size, callbacks=[custom_checkpoint])

model.save('yugi-model-melodi-predicter.tf')