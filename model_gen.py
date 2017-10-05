import csv
import cv2
import numpy as np

# Read the csv file which contains the training files, their steering angles.
lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

import sklearn
from sklearn.utils import shuffle

samples_augmented = 0
def generator(samples, batch_size=32):
    num_samples = len(samples)
    #print("num_samples", num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            count = 0
            for line in batch_samples:
                measurement = float(line[3])
                # Ignore runs of images with steering angle 0. Too much of bias on straight driving is eliminated.
                # also helps us on GPS instance with less memory needs.
                #Currently any run of images > 5 with steering angle 0 are ignored
                if (measurement == 0):
                    count = count + 1
                else:
                    count = 0
                if (count >= 5):
                    continue
                # Read the left, center and right images in each line.
                for i in range(3):
                    source_path = line[i]
                    source_path = source_path.strip()
                    local_path = "./data/" + source_path
                    image = cv2.imread(local_path)
                    image = np.array(image)
                    # Opencv reads the images in BGR format. Convert back to RGB.
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                # Add correction factor so that close to zero steering angle are rightly biased for left and right
                correction = 0.25
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)
            # trim image to only see section with road
            augmented_images = []
            augmented_measurements = []
            #print("images length", len(images))
            # Add augmented images so that training is done on reflected images too.
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = measurement * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)
            #print("augmented images length", len(augmented_images))
            X_train = np.array(augmented_images)
            samples_augmented 
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Build the neural net
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Build a LENET neural net.
model = Sequential()

# Normalize the image pixels to between [-0.5, 0.5]
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop the image so that sky, side mountains and upper portion and a
# little bottom portion are eliminated.
model.add(Cropping2D(cropping=((70,25), (0,0))))

#1st layer CNN, activate with RELU.
model.add(Convolution2D(6, 5, 5, activation='relu'))

# Max poolhelps in avoiding overfitting
model.add(MaxPooling2D())

# 2nd layer
model.add(Convolution2D(16, 5, 5, activation='relu'))

# Max poolhelps in avoiding overfitting
model.add(MaxPooling2D())

# Flatten the net
model.add(Flatten())

# Full layer
model.add(Dense(120))

# Full layer
model.add(Dense(84))

# Final layer
model.add(Dense(1))

# Optimize with adam gradient descendent version.
# This converges fast using larger steps but takes more computations.
# https://stats.stackexchange.com/questions/184448/difference-between-gradientdescentoptimizer-and-adamoptimizer-tensorflow
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples*3), validation_data=validation_generator,  nb_val_samples=len(validation_samples*3), nb_epoch=3)

model.save('model.h5')


    
