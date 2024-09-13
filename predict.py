import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model = tf.keras.models.load_model('fashion_aesthetic_model.h5')

# Define the image size and the class labels
IMG_SIZE = (224, 224)
labels = ['Boho', 'Fluffed Up', 'Granpa Chic', 'Hyper Feminine', 'Jelly Fashion', 'Leopard Print', 'Quiet Luxury']


def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    print(f"Predicted Fashion Aesthetic: {predicted_class}")

    # Load the image using PIL
    img_pil = Image.open(img_path)

    # Create an ImageDraw object to write text on the image
    draw = ImageDraw.Draw(img_pil)

    # Define the font size and style (use default if no custom font available)
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # For custom font, try arial or any available font
    except IOError:
        font = ImageFont.load_default()

    # Define text position and color
    text_position = (10,10 )  # Top-left corner
    text_color = (0, 0, 0)  # Red

    # Add the text to the image
    draw.text(text_position, predicted_class, fill=text_color, font=font)

    # Show the image with the label
    img_pil.show()


# Hardcoded image path
img_path = 'test/query_image.jpg'

# Call the prediction function
predict_image(img_path)
