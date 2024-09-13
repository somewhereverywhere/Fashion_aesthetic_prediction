from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from keras.layers import Dense , Flatten
from tensorflow.keras.optimizers import Adam
from architecture import x_train,y_train,x_val,y_val,datagen

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base VGG16 layers to prevent retraining
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  ])


model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    validation_data=(x_val, y_val),
                    epochs=100)
model.save('fashion_aesthetic_model.h5')

