import numpy as np
import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import io
import time

# Définir l'encodage de la sortie standard en UTF-8 (évite des crash de la console)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paramètres
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 15

# Générer les ensembles d'entraînement et de test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Redimensionner les données des pixels à l'intervalle [0, 1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Assure que les images ont la forme (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convertir les vecteurs de classes en matrices de classes binaires
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Définir le modèle
model = Sequential(
[
    layers.Input(shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

model.summary()

# Callback pour mesurer la durée de chaque époque
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.start_time = time.time()

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# Compiler et entraîner le modèle
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[time_callback])

# Save the model
model.save('model_cnn.keras')
print("Model saved to disk.")

# Sauvegarder le modèle
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Prédictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Rapport de classification
print("Classification report:")
print(classification_report(y_true, y_pred_classes))

# Matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.show()

# Visualisation de l'historique de l'entraînement
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Visualisation de l'historique de l'entraînement basé sur le temps
cumulative_times = np.cumsum(time_callback.times)

# Précision basée sur le temps
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cumulative_times, history.history['accuracy'], label='Training')
plt.plot(cumulative_times, history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy')
plt.legend()

# Perte basée sur le temps
plt.subplot(1, 2, 2)
plt.plot(cumulative_times, history.history['loss'], label='Training')
plt.plot(cumulative_times, history.history['val_loss'], label='Validation')
plt.title('Loss over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Loss')
plt.legend()

plt.show()
