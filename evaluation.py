from transfer import model
from architecture import y_val,x_val
from sklearn.metrics import classification_report
import numpy as np

# Get predicted probabilities
predictions = model.predict(x_val)

# Get the predicted class labels (using argmax to pick the class with the highest probability)
predicted_classes = np.argmax(predictions, axis=1)

# Use classification_report to evaluate the predictions
print(classification_report(y_val, predicted_classes,
      target_names=['Boho', 'Fluffed Up', 'Granpa Chic', 'Hyper Feminine', 'Jelly Fashion', 'Leopard Print', 'Quiet Luxury']))
