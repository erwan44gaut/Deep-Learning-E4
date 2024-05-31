from keras.models import load_model
import numpy as np

def multicouche(image):
    # Load the model
    model = load_model("./model_dense.keras")

    # Load and preprocess the image
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)
    image = image.astype("float32") / 255
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    prediction = np.argmax(predictions, axis=1)

    return str(prediction[0]), predictions[0]