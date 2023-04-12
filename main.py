import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_path):
    labels = []
    mfccs = []
    mels = []

    for filename in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, filename)):
            labels.append(str(filename))
            folder_path = os.path.join(data_path, filename)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                signal, sr = librosa.load(file_path, sr=44100)
                mfccs.append(librosa.feature.mfcc(y=signal, sr=sr))
                mels.append(librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000))
                
    return np.array(labels), np.array(mfccs), np.array(mels)


def visualize_data(mfccs, mels, fig, ax):
    img = librosa.display.specshow(librosa.power_to_db(mels[0], ref=np.max), x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()

    img = librosa.display.specshow(mfccs[0], x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')

def main():
    training_data_path = "datasets/IRMAS-Sample/Training/"

    labels, mfccs, mels = load_data(training_data_path)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    visualize_data(mfccs, mels, fig, ax)
    
    plt.show()


if __name__ == "__main__":
    main()
