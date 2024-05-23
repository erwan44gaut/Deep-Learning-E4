from keras.models import load_model
import numpy as np

def multicouche(image):
    # Load the model
    model = load_model("./model_dense.keras")

    # Load and preprocess the image
    image = image.resize((28, 28))  # Resize the image to 28x28
    image = image.convert('L')  # Convert image to grayscale
    image = np.array(image)
    image = image.astype("float32") / 255  # Normalize image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image)
    prediction = np.argmax(predictions, axis=1)
    return str(prediction[0])