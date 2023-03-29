import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split


def load_data(data_path, labels):
    data = []
    target = []
    
    
    for label in labels:
        folder_path = os.path.join(data_path, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            signal, sr = librosa.load(file_path, sr=44100)
            
            # Ekstrakcja cech MFCC
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            
            # Dodanie danych i etykiety do listy
            data.append(mfccs)
            target.append(label)
            
    return np.array(data), np.array(target)


def main():
    training_data_path = "datasets/IRMAS-Sample/Training"
    labels = ["sax", "vio"]

    
    X_train, y_train = load_data(training_data_path, labels)

    
    print("Training data size:", X_train.shape, y_train.shape)
    


if __name__ == "__main__":
    main()