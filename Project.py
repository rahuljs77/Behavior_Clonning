import csv
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda, Cropping2D, MaxPooling2D, Conv2D, Activation, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from math import ceil


#  STEER_LEFT = NEGATIVE!
epochs = 10
correction_factor = 0.3
batch_size = 70
input_size = 64
activation_relu = 'relu'

#  GENERATOR ------------------------------------------------------
# insert data generator code here
# DATA COLLECTION -------------------------------------------------
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering = []

for line in lines:
    for i in range(0, 3):
        source_path = line[i]
        file_name = source_path.split('\\')[-1]
        image = cv2.imread('IMG/' + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image_flipped = np.fliplr(image)
        steer = float(line[3])
        if i == 1:
            steer = steer + correction_factor
        elif i == 2:
            steer = steer - correction_factor

        images.append(image)
        images.append(image_flipped)
        steering.append(steer)
        steering.append(-steer)

X_train = np.array(images)
random_image = X_train[200]
print(random_image.shape)
y_train = np.array(steering)
# -----------------------------------------------------------------


def preprocess(image_yuv):
    image_yuv = cv2.cvtColor(image_yuv, cv2.COLOR_BGR2YUV)
    return image_yuv


# NETWORK ARCHITECTURE --------------------------------------------
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Lambda(lambda y: tf.image.resize(y, (66, 200))))
# # model.add(Lambda(lambda z: preprocess(z)))
# # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Conv2D(24, (5, 5), activation='relu'))
# model.add(Conv2D(36, (5, 5), activation='relu'))
# model.add(Conv2D(48, (5, 5), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Dropout(0.4))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(66, 200, 3))
for layer in vgg_model.layers:
    layer.trainable = False

data_input = Input(shape=(160, 320, 3))
resize_input = Lambda(lambda image: tf.image.resize(image, (66, 200)))(data_input)
vgg = vgg_model(resize_input)
drop_1 = Dropout(0.2)(vgg)
flat_1 = Flatten()(drop_1)
dense = Dense(512, activation='relu')(flat_1)
drop_2 = Dropout(0.2)(dense)
dense_2 = Dense(256, activation='relu')(drop_2)
drop_3 = Dropout(0.2)(dense_2)
dense_3 = Dense(64, activation='relu')(drop_3)
drop_4 = Dropout(0.2)(dense_3)
prediction = Dense(1)(drop_4)

#  ----------------------------------------------------------------
model = Model(inputs=data_input, outputs=prediction)
adam = Adam(lr=0.0001)
model.compile(optimizer='Adam', loss='mse')
print('training...')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, verbose=2, batch_size=batch_size)
# model.fit(train_generator, validation_data=validation_generator, verbose=2, epochs=epochs)
print()
print('training complete')
# , steps_per_epoch=ceil(len(train_lines)/batch_size) ....validation_steps=ceil(len(validation_lines)/batch_size)

model.save('model.h5')
print()
print('model saved')

# lines = []
# with open('driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)
#
# train_lines, validation_lines = train_test_split(lines, test_size=0.2)
# # test_line = lines[0]
# # center_name = 'IMG/' + test_line[0].split('\\')[-1]
# # center_image = cv2.imread(center_name)
# # plt.imshow(center_image)
# # plt.show()
# # flipped_image = np.copy(center_image)
# # flipped_image = np.fliplr(flipped_image)
# # print(flipped_image)
#
#
# def generator(samples, batch_size):
#     num_samples = len(samples)
#     while 1:  # Loop forever so the generator never terminates
#         # shuffle(lines)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset + batch_size]
#
#             images = []
#             steering = []
#             for batch_sample in batch_samples:
#                 center_name = 'IMG/' + batch_sample[0].split('\\')[-1]
#                 left_name = 'IMG/' + batch_sample[1].split('\\')[-1]
#                 right_name = 'IMG/' + batch_sample[2].split('\\')[-1]
#
#                 center_image = cv2.imread(center_name)
#                 flipped_image = np.copy(center_image)
#                 flipped_image = np.fliplr(flipped_image)
#                 left_image = cv2.imread(left_name)
#                 right_image = cv2.imread(right_name)
#                 print(right_image.shape)
#
#                 center_angle = float(batch_sample[3])
#                 flipped_angle = -center_angle
#                 left_angle = center_angle + correction_factor
#                 right_angle = center_angle - correction_factor
#
#                 images.append(center_image)
#                 images.append(flipped_image)
#                 images.append(left_image)
#                 images.append(right_image)
#
#                 steering.append(center_angle)
#                 steering.append(flipped_angle)
#                 steering.append(left_angle)
#                 steering.append(right_angle)
#
#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(steering)
#             yield sklearn.utils.shuffle(X_train, y_train)
#
# train_generator = generator(train_lines, batch_size)
# validation_generator = generator(validation_lines, batch_size)









