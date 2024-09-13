**Overview**

This project aims to predict 2024 fall/winter fashion aesthetics using a machine learning model based on image data. We leverage transfer learning by fine-tuning the VGG16 pre-trained model, enabling us to achieve a promising level of accuracy even with a relatively small dataset.

Results after 100 epoch:

Accuracy: 0.74 (74%)

Macro Avg Precision: 0.76

Macro Avg Recall: 0.75

Macro Avg F1-Score: 0.73

Weighted Avg Precision: 0.79

Weighted Avg Recall: 0.74

Weighted Avg F1-Score: 0.74

These metrics were calculated using a classification report from the model's performance on the test dataset, demonstrating the model’s ability to generalize across various fashion aesthetics.

**Model Architecture**

We used VGG16, a popular convolutional neural network pre-trained on ImageNet, as the base for transfer learning. By fine-tuning the model, we could leverage the powerful feature extraction capability of VGG16 while adapting it to our specific task of fashion aesthetic prediction.

Transfer Learning

VGG16 Pre-trained Model: The model has been pre-trained on millions of images and can extract high-level features, making it suitable for image classification tasks.

Fine-tuning: Only the final layers of the VGG16 model were retrained to adapt to the fashion aesthetic categories in our dataset.

**Dataset**

Due to the small size of the dataset, transfer learning was crucial in ensuring the model performs well. The dataset consists of fashion images labeled with different aesthetic categories (e.g., casual, formal, sporty, etc.).

Total images:584

training:450

validation:134

Classes: ['Boho','Fluffed Up','Granpa Chic','Hyper Feminine','Jelly Fashion','Leopard Print','Quiet Luxury']

**Future Work**

Larger Dataset: With more data, we expect the model's performance to improve significantly.

Model Architecture Tuning: Further fine-tuning or experimenting with other architectures (e.g., ResNet, Inception) might yield better results.

Real-time Predictions: Implementing a real-time prediction system for applications in fashion retail.

**Conclusion**

Despite the limited dataset, this project demonstrates the potential of transfer learning with VGG16 in predicting fashion aesthetics. The model achieved a reasonable level of accuracy and generalization across different categories, and future improvements could further enhance its performance.

Feel free to contribute to this project by exploring different models, optimizing hyperparameters, or adding new datasets!
