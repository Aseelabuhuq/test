from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import cv2
import os
import argparse
import sys
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from keras.preprocessing import image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import keras
import tensorflow as tf

images= os.listdir("./train/train")
#print(images)
data = pd.DataFrame(images)
data = data.rename(columns={0: "image"})
data['image'] = data['image'].apply(lambda x: "./train/train/"+x)
data['label'] = data['image'].apply(lambda x: 0 if 'cat' in x else 1)
print(data.head(19500))


# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
train_image = []
for i in tqdm(range(data.shape[0])):
    img = keras.utils.load_img(data['image'][i], target_size=(64,64,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


y=data['label'].values
print(y)

####################################################


x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.1)

print('x tain is:', x_train.shape, 'x valid is', x_valid.shape)
print('y tain is:', y_train.shape, 'y valid is', y_valid.shape)


##############################################
model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) # 2 because we have cat and dog classes

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

############################################################

filepath = 'firstmodel.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                                   verbose=1, mode='auto', epsilon=0.0001)

earlystop = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=7)

callbacks_list = [checkpoint, reduceLROnPlat, earlystop]

#########################################################

callbacks_list = [checkpoint, reduceLROnPlat, earlystop]

history = model.fit(x_train, np.asarray(y_train),
                    callbacks=callbacks_list,
                    validation_data=(x_valid, np.asarray(y_valid)),
                    epochs=10,
                    batch_size=256)
                    

#######################################################

fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(history.epoch, history.history["binary_accuracy"], label="Train ACC")
ax[1].plot(history.epoch, history.history["val_binary_accuracy"], label="Validation ACC")
ax[0].legend()
ax[1].legend()
plt.show()
y_pred = model.predict(x_train)

print(classification_report(y_train.argmax(axis=1), y_pred.argmax(axis=1)))

y_pred = model.predict(x_valid)
print(classification_report(y_valid.argmax(axis=1), y_pred.argmax(axis=1)))