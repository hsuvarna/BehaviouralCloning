import csv
import cv2
import numpy as np

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print(lines[1])

images = []
measurements = []
count = 0
for line in lines[1:]:
    measurement = float(line[3])
    if (measurement == 0):
        count = count + 1
    else:
        count = 0
    if (count >= 5):
        continue
    for i in range(3):
        source_path = line[i]
        source_path = source_path.strip()
        local_path = "./data/" + source_path
        #print(local_path)
        image = cv2.imread(local_path)
        #height = np.size(image, 0)
        #width = np.size(image, 1)
        #print(height, width)
        images.append(image) 
    correction = 0.25
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

print(len(images), len(measurements))
X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)
print("shape is", X_train.shape)

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, nb_epoch=3, validation_split=0.2, shuffle=True)

model.save('model.h5')


    
