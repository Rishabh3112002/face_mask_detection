import numpy as np
import PIL
import tensorflow as tf
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

dir = r'PATH_TO_DATASET'
cat = ["with_mask", "without_mask", "mask_weared_incorrect"]#folder name of the dataset

print("[INFO] loading images...")

data = []
labels = []

for c in cat:
    path = os.path.join(dir, c)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        labels.append(c)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

data = np.array(data, dtype='float32')
label = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels)

print(trainX.shape)
print(testX.shape)

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(256, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print('Model Compiling...')
opt = Adam(lr=1e-4, decay=1e-4/10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(trainX, trainY, batch_size=32, epochs=10)

pred = model.predict(testX, batch_size=10)

pred = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1), pred, target_names=le.classes_))

print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

N = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, N), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, N), H.history['accuracy'], label="train_acc")
plt.plot(np.arange(0, N), H.history['val_accuracy'], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
