import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ----------------------------
# 1. Dataset Setup
# ----------------------------
base_dir = "C:/Users/DELL/OneDrive/Desktop/AI&Ml internship/Week 1 Tasks/Task 2/dataset"  
img_size = (128, 128)
batch_size = 32

# ----------------------------
# 2. Data Generators
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print(f"✅ Train samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

# ----------------------------
# 3. Build Transfer Learning Model
# ----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128,128,3))
base_model.trainable = False   # freeze base model

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ----------------------------
# 4. Train Model
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# ----------------------------
# 5. Plot Accuracy/Loss
# ----------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure()
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.show()

# ----------------------------
# 6. Confusion Matrix & Report
# ----------------------------
val_preds = (model.predict(val_gen) > 0.5).astype("int32")
cm = confusion_matrix(val_gen.classes, val_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat","Dog"], yticklabels=["Cat","Dog"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(val_gen.classes, val_preds, target_names=["Cat","Dog"]))

# ----------------------------
# 7. Prediction on New Image
# ----------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]
    if pred > 0.5:
        print(f"🦮 {os.path.basename(img_path)} -> Dog ({pred:.2f})")
    else:
        print(f"🐱 {os.path.basename(img_path)} -> Cat ({1-pred:.2f})")

# Example usage:
# predict_image("C:/Users/DELL/OneDrive/Desktop/AI&Ml internship/Week 1 Tasks/Task 2/dataset/cats/cat_20.jpg")
# predict_image("C:/Users/DELL/OneDrive/Desktop/AI&Ml internship/Week 1 Tasks/Task 2/dataset/dogs/dog_20.jpg")

# ----------------------------
# 8. Interactive Testing
# ----------------------------
print("\n✅ Training finished. You can now test images!")
print("Type the path of an image (e.g. dataset/cats/cat_20.jpg)")
print("Type 'exit' to quit.")

while True:
    path = input("\nEnter image path: ")
    if path.lower() == "exit":
        break
    if os.path.exists(path):
        predict_image(path)
    else:
        print("⚠️ File not found, try again.")
