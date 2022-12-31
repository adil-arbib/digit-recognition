import os
import cv2
import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
import pygame


# mnist = tf.keras.datasets.mnist
# (x_train, y_train) , (x_test, y_test) = mnist.load_data()
#
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(100,activation='softmax'))
#
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)
#
# model.save('dajdadmodel')

model = tf.keras.models.load_model('dajdadmodel')

img_num = 1

while os.path.isfile(f"digits_img/digit{img_num}.png"):
    try:
        img = cv2.imread(f"digits_img/digit{img_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"the number is  {np.argmax(prediction)}")
        pl.imshow(img[0], cmap=pl.cm.binary)
        pl.show()
    except:
        print("Error !!")
    finally:
        img_num += 1