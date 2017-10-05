#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/img5000.jpg  "Normal training image"
[image2]: ./examples/cropped_img5000.jpg "Cropped training image. Borders blacked"
[image3]: ./examples/flipped_img5000.jpg "Flipped training image"
[image4]: ./examples/img7000.jpg  "Normal training image"
[image5]: ./examples/cropped_img7000.jpg "Cropped training image. Borders blacked"
[image6]: ./examples/flipped_img7000.jpg "Flipped training image"

## Rubric Points

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run2.mp4 - Video I recorded for lap1 testing with model5.h
* run1.mp4 - Another video I recorded.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network as described in Lenet. Per the Video instructions in David Silvan video, I first created a simple neural network. It has 2 convolution layers of 5x5 with varying depths (6, 16) and followed by full layers of 120, 84 and 1. I also used Pooling.

I also tried the nvidea neural net model (see the model_nvidea.py). But I could not get good results out of that.

I also tried various models by varying the correction factor all the way from 0.2 to 0.33 with 0.2 intervals. 0.25 value gave better results.

####2. Attempts to reduce overfitting in the model

Max pooling is employed to reduce the overfitting. Also other image processing techniques like generating flipped images ensured that the program is trained on different sets of road conditions.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I tried epochs 3 and 5. After 3 epochs, the validation error plateoed.So made it 3 epochs and that also saved time on AWS.

Other than this parameter, I did not try any other parameter.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road using correction factor of 0.25. The flipped images also helped here. In addition, cropping the images reduced the effects of mountains, sky and water in images.

I used the sample project training data ![alt text][image1] ![alt text][image4]. I filtered out zero steering angle images if more than 5 in sequentially.


###Model Architecture and Training Strategy

####1. Solution Design Approach

Tried countless times to stratighten the car on the road. 

* Tried the Lenet and nvidea neural nets.
* Tried zero steering angle image filtering if sequence > 3 or 4 or 5.
* Tried correction factor of 0.2 to 0.33 with 0.2 intervals.
* added extra dropout layers but did nt help.
* Used 3 cameras.
* Used flipped images ![alt text][image3] 
![alt text][image6] 
* Augmented extra images

####2. Final Model Architecture

Final model architecture is Lenet architecture. Used RELU as activation function. Per https://stats.stackexchange.com/questions/218752/relu-vs-sigmoid-vs-softmax-as-hidden-layer-neurons
The Relu functions better for neural nets with fast convergence.

Max pooling is used for reducing overfitting.

####3. Creation of the Training Set & Training Process

* Training data: I used from the Project resources
* Preprocessed the images tobe with in -0.5 to 0.5.


