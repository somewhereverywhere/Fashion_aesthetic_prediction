
import cv2
import os

import numpy as np

labels = ['Boho','Fluffed Up','Granpa Chic','Hyper Feminine','Jelly Fashion','Leopard Print','Quiet Luxury']
img_size = 224
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]  # Convert BGR to RGB
                if img_arr is None:  # Check if image was loaded successfully
                    print(f"Failed to load image: {img}")
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Resize image
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing image {img}: {e}")
    return np.array(data, dtype=object)  # Use dtype=object to avoid the error


train = get_data('train')
val = get_data('validate')

