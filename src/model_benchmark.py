from keras.models import load_model
import numpy as np
from keras.datasets import mnist
import time
import sys
import io

# Définir l'encodage de la sortie standard en UTF-8 (évite des crash de la console)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Benchmarking the CNN model...")
# Load the model to be benchmarked
model = load_model("./model_cnn.keras")

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess the images
x_test = x_test.astype("float32") / 255 
x_test = np.expand_dims(x_test, axis=-1)
x_test = np.expand_dims(x_test, axis=1)
x_test = np.squeeze(x_test)

# Benchmark prediction time
start_time = time.time()

# Make predictions
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

end_time = time.time()

# Calculate and print the total prediction time
total_time = end_time - start_time

print(f"Total prediction time for {len(x_test)} images: {total_time:.4f} seconds")
print(f"Average prediction time per image: {total_time / len(x_test):.6f} seconds")

print("Benchmarking the multilayer model...")
# Load the model to be benchmarked
model = load_model("./model_dense.keras")

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess the images
x_test = x_test.astype("float32") / 255 
x_test = np.expand_dims(x_test, axis=-1)
x_test = np.expand_dims(x_test, axis=1)
x_test = np.squeeze(x_test)

# Benchmark prediction time
print("Starting benchmark...")
start_time = time.time()

# Make predictions
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

end_time = time.time()

# Calculate and print the total prediction time
total_time = end_time - start_time

print(f"Total prediction time for {len(x_test)} images: {total_time:.4f} seconds")
print(f"Average prediction time per image: {total_time / len(x_test):.6f} seconds")
