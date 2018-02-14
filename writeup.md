# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

This is the second project in the Self-Driving Car Engineer Nanodegree programme. The steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images_for_writeup/TrainingSet.png "Original image distribution"
[image2]: ./images_for_writeup/TrainingSetAugmented.png "Augmented image distribution"
[image3]: ./images_for_writeup/Grayscaling.png "Gray scaling of image"
[image4]: ./images_for_writeup/RotatedImage.png "Rotation of image"
[image5]: ./images_for_writeup/BlurredImage.png "Blurring of image"
[image6]: ./images_for_writeup/TranslatedImage.png "Translation of image"
[image7]: ./images_for_writeup/tsign1.png "German traffic sign 1"
[image8]: ./images_for_writeup/tsign2.png "German traffic sign 2"
[image9]: ./images_for_writeup/tsign3.png "German traffic sign 3"
[image10]: ./images_for_writeup/tsign4.png "German traffic sign 4"
[image11]: ./images_for_writeup/tsign5.png "German traffic sign 5"

## Rubric Points
In the following I will go through the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

In general this writeup is submitted together with the jupyter notebook, so please refer to code which is available both as ipynp and in html form.

---
### Writeup / README

As a starting remark it is important to note that I submitted the project as soon as the results were "good enough". This is due to time constraints on my part. It would therefore very likely have been possible to achieve better results with my model. Also I did not do the optional part. Having said that the model is "good enough", showing a validation accuracy of 0.931, just above the required 0.93.

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set. The results for the original data set were:

* The size of training set is 34799 images
* The size of the validation set is 12630 images
* The size of test set is 4410 images
* The shape of a traffic sign image is 32 x 32 x 3, equal to 32 by 32 pixels, each with a value of r,g,b
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The following bar chart shows the distribution of the training set images over the 43 classes. It can be seen that there is a very varying number of images available. The mean number of images is 809, but there are many classes with a lot less images. Therefore I will (see below) generate additional training images.  

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Due to the distribution of number of images per class I decided to generate more training data. The method is to generate variations of the existing training images, by applying various image processing operations. There are many possibilities and I decided (more or less arbitrarily) to go for the following 3:

* Image rotation.
* Image blurring.
* Image translation.

In each of these operations I applied some kind of randomization:

* Images were rotated around their center with a random angle between -45 and + 45 degrees.
* Images were blurred using gaussian blur with a random kernel size between (1,1) and (5,5).
* Images were shifted in the x and y direction with random values in the range of -0.3*32 t0 0.3*32 (32 being the original image size).

For all 3 operations I used the OpenCV library which provides all necessary methods. For code reference please look for the functions rotate_image(), blur_image() and translate_image().

The rotation operation does something like this:

![alt text][image4]

The blurring operation does something like this:

![alt text][image5]

The translation operation does something like this:

![alt text][image6]

I applied the operations in the training image classes with less than 1000 images. The operations were applied to random selections of the original images until each class had at least 999 images. This number is also chosen more or less arbitrarily. It is clear that the more images per training class the better.

After generating new images the distribution of training images looks like:

![alt text][image2]

In the next step I translated the images to being gray scale, as this is required in the project. Gray scaling was done using the OpenCV convert color method, in the functions convert_to_gray() and make_them_gray().

It results in changes like this:

![alt text][image3]

Finally the project requires normalization of the grayscaled images so the pixel values are in the range 0.1 to 0.9. The course material explains this as helping to keep numerical stability and also provides the formulate for changing the pixel values:

x' = a + (x - xmin)(b - a)/(xmax - xmin)

where a = new min, b = new max, xmin = original min, xmax = original max

Normalization is implemented in the function normalize().

Gray scaling and normalization must be applied to all 3 sets of images: train, test and validation, as they must all have the same format to fit with the model format.

The total result of my preprocessing was that I had:

51662 training images, of the shape (32, 32, 1)
12630 validation images, of the shape (32, 32, 1)
4410 test images, of the shape (32, 32, 1)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In fact I implemented 2 different neural networks, both variants of the LeNet architecture. The first model was my network from the LeNet-Lab modified to have 43 outputs, corresponding to the number of different classes we want to identify. This model architecture can be summarized as:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray scaled image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input 400 output 120        									|
| RELU					|												|
| Fully connected		| input 120 output 84        									|
| RELU					|												|
| Output Fully connected		| input 84 output 43        									|

As explained in the project material this architecture was just the starting point and would not produce the required performance. It didn't.

So I decided to go for the real thing and implement the architecture described the Pierre Sermanet and Yann LeCun in their paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" which was referenced by the project material. This architecture looks like this in my implementation:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray scaled image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Concatenation of L2 and L3 outputs              | input 5x5x16 and 1x1x400, output 800                      |
| Dropout layer              |                       |
| Output Fully connected		| input 800 output 43        									|

It shall be said that the paper does not describe the architecture in detail, so the above is my interpretation of what they wrote. I used that model for remainder of the project and you can find my implementation in the function LeNetFromArticle().

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the setup from the LeNet-Lab which used the AdamOptimizer. My final solution before submission used the following hyper parameters:

* Batch size = 128
* Number of epochs = 60
* Mu = 0 (for weight initialization)
* Sigma = 0.1 (for weight initialization)
* Learning rate = 0.0005
* Keep probability = 0.5 (for the dropout layer)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My model results before project submission were:
* Training set accuracy = 0.999
* Validation set accuracy of 0.931
* Test set accuracy of 0.931

The validation set accuracy was calculated at the end of each epoch, and the result 0.931 was the value after epoch number 60.
The test set accuracy was calculated as the very last thing after I had decided to submit the project, as my validation set accuracy was above the minimum requirement of the project.

The approach for finding the solution was in the end semi-iterative. First I tried (as recommended) my architecture from the LeNet-Lab in the course, and found (as expected) that this accuracy of this was not quite good enough for solving the project.
Rather than try different architectures I went for "the real thing" directly, namely the architecture of Pierre Sermanet and Yann LeCun, in the expectation/hope that my understanding of their architecture would be good enough to solve the problem at hand. Indeed it was.

My accuracy results in the end are not nearly as good as what the masters achieved. Partly I think this is because I decided to submit the project as soon as my results were "good enough". I think it would be possible to achieve better results by spending more time tuning the hyper parameters. In particular the learning rate seemed to have a real impact. Further attempts, had I continued, would have been to reduce the learning rate dynamically during model training (e.g. 10% every 10 epochs), and to try with more epochs.

Another reason for my "not that impressive" accuracy is, I believe, the augmentation of the training data I did. The article describes Sermanet and LeCun adding image jitter, and my thinking is that this is a better form of augmentation that the largely geometrical operations I applied. The difference between the training set accuracy and the validation set accuracy hints at overfitting which is funny given I am using the standard model. Again this points me to my training data which in all likelihood should be augmented in different ways and with more "randomness".

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

The images have been resized to 32x32 already.

The images are all pretty clear and I would expect my neural network to be able to classify all of them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Below are the results of the prediction. The name of each image is taken from the provided signnames.csv file.

#### Image 1: Speed Limit (20 km/h)

Logits: [35.61962   19.815168  19.163353  10.620274  -2.2068405]

Probabilities: [9.99999762e-01 1.36840043e-07 7.13071984e-08 1.38970025e-11
 3.73401670e-17]

Prediction: [0 1 4 5 2], Class 0 is the correct prediction

#### Image 2: Ahead only

Logits: [29.522577   3.2078102  0.926318  -1.8230546 -2.0403922]

Probabilities: [1.0000000e+00 3.7294222e-12 3.8089218e-13 2.4364906e-14 1.9605442e-14]

Prediction: [35 36 33 34 14], Class 35 is the correct prediction

#### Image 3: Pedestrians

Logits: [45.722176  22.524273  15.514791  13.04162    7.1491427]

Probabilities: [1.0000000e+00 8.4193562e-11 7.6049952e-14 6.4123071e-15 1.7698843e-17]

Prediction: [27 18 11 24 26], Class 27 is the correct prediction

#### Image 4: Dangerous curve to the right

Logits: [18.083492    6.182893    4.7831187  -0.79051965 -1.7283148 ]

Probabilities: [9.9999154e-01 6.7862779e-06 1.6738544e-06 6.3550223e-09 2.4879268e-09]

Prediction: [20 23 28 30 24], Class 20 is the correct prediction

#### Image 5: No passing

Logits: [16.213955   3.4726624  1.5099156 -2.927869  -3.3150353]

Probabilities: [9.9999666e-01 2.9276916e-06 4.1125890e-07 4.8619411e-09 3.3011558e-09]

Prediction: [ 9 10 41 23 34], Class 9 is the correct prediction

The model was able to correctly guess all 5 of the traffic signs, which gives an accuracy of 100%. You can see on the values of the top 5 logits and probabilities that the network in all 5 cases was very convinced about its prediction.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As it can be see from the listing in the above section the model is very certain of its predictions. In all 5 cases it gives a probability very close to 1 as its top value.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I did not do the optional part.
