from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RepeatedKFold
import matplotlib.pyplot as plt


def load_audio_files(data_dir, instruments, features_type):
    X = []
    y = []
    counter = 0
    for instrument in instruments:
        instrument_dir = os.path.join(data_dir, instrument)
        for filename in os.listdir(instrument_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(instrument_dir, filename)
                audio, sr = librosa.load(filepath, sr=None)
                if features_type == 'mfcc':
                    feature = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
                elif features_type == 'mel':
                    feature = librosa.feature.melspectrogram(y=audio, sr=sr)
                X.append(feature.T)
                y.append(instrument)
                counter += 1
                if counter == 5:
                    break
    X = np.array(X)
    y = np.array(y)
    return X, y


def preprocess_labels(y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y = onehot_encoder.fit_transform(integer_encoded)
    return y


def create_model(input_shape, num_outputs, with_dropout):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
              input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if with_dropout:
        model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if with_dropout:
        model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if with_dropout:
        model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    if with_dropout:
        model.add(Dropout(0.3))
    model.add(Dense(num_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X_train, X_val, y_train, y_val, batch_size, epochs):
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_data=(X_val, y_val))
    return history


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def log_stats(title, lose, accuracy):
    file = open('logs.log', 'a+')
    file.write(title + '\n')
    file.write('Lose ')
    file.write(str(lose) + '\n')
    file.write('Accuracy ')
    file.write(str(accuracy) + '\n')
    file.close()


def experiment(X, y, instruments, with_dropout, epochs, title):
    print(title)
    rkf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=123)
    lose = []
    accuracy = []
    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
        model = create_model(
            (X_train.shape[1], X_train.shape[2], 1), num_outputs=len(instruments), with_dropout=with_dropout)
        history = train_model(model, X_train, X_val, y_train, y_val, 32, epochs)

        score = model.evaluate(X_val, y_val, verbose=0)
        lose.append(score[0])
        accuracy.append(score[1])
    log_stats(title, lose, accuracy)


def main():
    data_dir = 'datasets/IRMAS-TrainingData'
    instruments = ['cel', 'cla', 'flu']

    X, y = load_audio_files(data_dir, instruments, 'mel')
    y = preprocess_labels(y)
    experiment(X, y, instruments, True, 15, 'MEL + droput')
    experiment(X, y, instruments, False, 15, 'MEL - droput')

    X, y = load_audio_files(data_dir, instruments, 'mfcc')
    y = preprocess_labels(y)
    experiment(X, y, instruments, True, 15, 'MFCC + droput')
    experiment(X, y, instruments, False, 15, 'MFCC - droput')
    experiment(X, y, instruments, True, 5, 'MFCC 5 epok')
    experiment(X, y, instruments, True, 10, 'MFCC 10 epok')
    experiment(X, y, instruments, True, 20, 'MFCC 20 epok')


if __name__ == "__main__":
    main()
