from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import matplotlib.pyplot as plt
import os

print("Starting script...")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    "data/train",
    target_size=(256,256),
    color_mode="grayscale",
    class_mode="binary"
)

print("Data loaded...")

model = build_model()

print("Model built...")

history = model.fit(
    train_data,
    steps_per_epoch=50,
    epochs=2
)

print("Training finished...")

model.save("models/model.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Performance')
plt.legend(['Accuracy', 'Loss'])
plt.savefig('outputs/training_graph.png')
plt.show()