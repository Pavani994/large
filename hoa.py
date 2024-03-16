import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Load the trained model
loaded_model = tf.keras.models.load_model('C:\\Users\\para pavani\\Downloads\\VGG16.h5')

# Class names obtained from the generate_dataset function
# Replace the next line with the actual code to generate class_names
class_names = ['breast_benign', 'breast_malignant']

def classify_image(img):
    # Resize the image to the required target size
    img = img.resize((224, 224))
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the class label
    predicted_class = class_names[predicted_class_index]

    return predicted_class

# Gradio app
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    live=True,
    title="Breast Cancer Image Integrating Horse Optimization Algorithm	",
    description="Upload an image and get the predicted class.",
)

iface.launch()
