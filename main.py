from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import os

train_img = ImageDataGenerator(rescale=1./255,fill_mode='nearest')

test_img = ImageDataGenerator(rescale=1./255)

train_generator = train_img.flow_from_directory(os.path.join("/TrainData"),target_size=(150,150),batch_size=64,class_mode='binary')

test_generator = test_img.flow_from_directory(os.path.join("/TrestData"),target_size=(150,150),batch_size=64,class_mode='binary')

plt.imshow(train_generator[0][0][5])
print("Label : ",train_generator[0][1][5])

plt.imshow(train_generator[0][0][60])
print("Label : ",train_generator[0][1][60])
