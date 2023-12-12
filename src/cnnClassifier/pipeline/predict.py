import numpy as np
from keras.models import load_model
# from keras.preprocessing import image
import os
from PIL import Image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "trained_model.keras"))

        # imagename = self.filename

        image_path = self.filename  # Replace with the path to your image

        img_size = (224, 224)  # Same size as specified in the generator

        single_test_image = Image.open(image_path)

        # Resize the image to match the input size expected by the model
        img_size = (224, 224)  # Use the same size as defined in the generators
        processed_image = single_test_image.resize(img_size)

        # Convert the image to an array
        processed_image = np.array(processed_image) / 255.0  # Normalize pixel values if required

        # Reshape the image to match the input shape expected by the model
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

        result = model.predict(processed_image)

        print(result)

        if result[0][0] > 0.5:
            prediction = 'Healthy'
            return {"result": prediction}
        else:
            prediction = 'Coccidiosis'
            return {"result": prediction}