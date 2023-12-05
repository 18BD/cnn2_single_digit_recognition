import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Dropout


class Model:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_data(self, path):
        files = librosa.util.find_files(path, ext=['wav'])
        files = np.asarray(files)

        data = []
        x = []
        y = []

        for file in files:
            linear_data, _ = librosa.load(file, sr=16000, mono=True)
            data.append(linear_data)
            y.append(file[37])

        for i in range(len(data)):
            x.append(abs(librosa.stft(data[i]).mean(axis=1).T))
        x = np.array(x)
        x = x.reshape(x.shape[0], x.shape[1], 1)

        y = np.array(y)
        y = y.reshape(-1)
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25)

        return x_train, x_test, y_train, y_test

    def build_model(self):
        self.model = Sequential()
        self.model.add(Convolution1D(filters=128, kernel_size=6,
                    activation='relu', input_shape=(1025, 1)))
        self.model.add(Convolution1D(
            filters=128, kernel_size=6, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution1D(
            filters=128, kernel_size=6, activation='relu'))
        self.model.add(Convolution1D(
            filters=128, kernel_size=6, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=10, activation='relu'))
        self.model.add(Dense(units=10, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def train(self, x_train, y_train, epochs=30, batch_size=16, validation_split=0.1):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def evaluate(self, x_test, y_test, batch_size=16):
        _, accuracy = self.model.evaluate(
            x_test, y_test, batch_size=batch_size)
        return accuracy

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, path):
        self.model.save(path)

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

path = "dataset\\"
cnn_model = Model()

x_train, x_test, y_train, y_test = cnn_model.load_data(path)
cnn_model.build_model()
history = cnn_model.train(x_train, y_train, epochs=30,batch_size=16, validation_split=0.1)
accuracy = cnn_model.evaluate(x_test, y_test, batch_size=16)
print(f'Accuracy: {round(accuracy*100, 2)} %')

cnn_model.plot_accuracy(history)
y_pred = cnn_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
class_labels = np.argmax(y_test, axis=1)
unique_labels = np.unique(np.concatenate((y_test, y_pred))).astype(int)
classes = cnn_model.label_encoder.inverse_transform(unique_labels)
cnn_model.plot_confusion_matrix(y_test, y_pred, classes)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classification_report(class_labels, y_pred, class_names))

cnn_model.save_model('model.h5')
