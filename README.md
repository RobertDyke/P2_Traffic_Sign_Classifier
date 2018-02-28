#**Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images



[//]: # (Image References)

[image1]:  hister_graph.png
[image2]:  gray.png
[image3]:  gray_normalized.png
[image4]:  blue.png
[image5]:  green.png
[image6]:  red.png
[image7]:  Traffic_Signs.png
[image8]: ./examples/placeholder.png "Traffic Sign 5"



0.In Step 0 I've loaded the data from the traffic_signs_data file, broken it into the three parts and examined the size and shape of the files. The three parts are training, validation, and testing.

X_train.shape :  (34799, 32, 32, 3)
X_test.shape :  (12630, 32, 32, 3)
y_test.shape :  (12630,)
X_valid.shape :  (4410, 32, 32, 3)

1.In Step 1 I've used python and numpy to create a data summary. There are 34799 training examples, 12630 testing examples, and all examples are (32,32,3).

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

![alt text][image1] (Markdown does not like my images. See histogram entitled "Distribution of 43 sign types" in code.)

2.I've explored using several python tools to display the data. Matplotlib's subplot works best, subplot2grid's arguments are odd. I've also broken out the 43 individual labels with an example of each.

(See code output as too many images to drag to markdown file.)

###Design and Test a Model Architecture

####1. I started out using LeNet architecture. This required changing the (32,32,3) image to (32,32,1). Thus, I converted the color images to grey scale. The literature I found on this strongly suggested normalizing the data once converted to gray scale. Some examples are printed below.

![alt text][image2]

![alt text][image3]

####1.5.After converting the data to gray scale I wondered what would happen if I striped out the red, green, and blue components of the color file. This would create three data sets meeting the (32,32,1) requirement for LeNet. The code is below. In the gray scale data I normalized it using a built in cv2 function. I see no similar function for red, green, and blue. So I am trying these three cases with no normalization. This is in cell 5 of the code.

Blue
![alt text][image4]

Green
![alt text][image5]

Red
![alt text][image6]

####2. LeNet was used in a prior lesson so I started with it here. Later I converted it to Sermanet-LeNet after reading one of the recommended papers. I was interested in two parts of the paper. One was splitting the data into two parts; one following the entire path and the second skipping several steps and remerging. The other part was a comment in the paper claiming that Sermant-LeNet was, in part, reverse engineered from cat and monkey visual cortexes. How cool is that? This is in cell 6 of the code. (paper url: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) (other useful paper:http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) The layers described for Sermanet-LeNet are illustrated in the seceond paper.


Layers of Sermanet-LeNet

input 32,32,1 
image convolution 1 output 28x28x6 
activation one pooling 1 output 14x14x6 
convolution 2 output 10x10x16 
activation 2 
pooling 2 output 5x5x16 
convolution 2B output 1x1x400 
activation 3 
flatten output of pooling 2 to output of 400 
flatten output of convolution 2B to output of 400 
concatenate the prior two outputs producting an output of 800 
truncate 800 to 43




####4. Again I used the code structure found in the prior lesson. This includes the AdamOptimizer. As I found out if the AdamOptimizer if left out (a typo error on my part). This leads to quite poor performance. A major modification was to change my tensorflow kernel to tensorflow-gpu. This cut run times into less than a quarter of what they were before. This is in cell 7 of the code.



####5.My training, testing, and validation sets are listed below. The best training set accuracy was 93.6% and validation was 91.4% this was with the normalized gray scale images. The best test set accuracy was only 86.1%. This was probalby due to the low number of training images on many of the 43 classes. This could have been improved by generating more data. My results are included in comments in the code organized by case, epochs, and batch size. The red, green, and blue cases were not normalized as discussed prior. I have included an un-normalized gray case for comparison purposes. The most interesting result is that the green case consistintly beat the un-normalized gray scale case. If I had more time, I would write code to normalize the green case and see if it would beat gray scale.  

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 93.6%
* validation set accuracy of 91.4%
* test set accuracy of 86.1%

Gray case not normalized
 40 Epochs Batch size 128  Validation Accuracy 81.6% Validation Accuracy 77.3% 
 10 Epochs Batch size 1024 Validation Accuracy 62.8% Validation Accuracy 60.7%
 20 Epochs Batch size 2048 Validation Accuracy 70.7% Validation Accuracy 70.1%


Gray case normalized
 40 Epochs Batch size 128  Training Accuracy 93.6% Validation Accuracy 91.4% 
 10 Epochs Batch size 1024 Training Accuracy 87.1% Validation Accuracy 91.4%
 20 Epochs Batch size 2048 Training Accuracy 87.7% Validation Accuracy 85.6%

 Blue case
 40 Epochs Batch size 128  Training Accuracy 87.3% Validation Accuracy 86.5% 
 10 Epochs Batch size 1024 Training Accuracy 76.9% Validation Accuracy 76.4%
 20 Epochs Batch size 2048 Training Accuracy 77.3% Validation Accuracy 78.6%

 Green case
 40 Epochs Batch size 128  Training Accuracy 90.7%  Validation Accuracy 87.5% 
 10 Epochs Batch size 1024 Training Accuracy 82.5% Validation Accuracy 81.7%
 20 Epochs Batch size 2048 Training Accuracy 77.0% Validation Accuracy 75.4%

 Red case   
40 Epochs Batch size 128  Training Accuracy 86.0%  Validation Accuracy 83.6%
10 Epochs Batch size 1024 Training Accuracy 76.5%  Validation Accuracy
20 Epochs Batch size 2048 Training Accuracy 79.7%  Validation Accuracy 71.6%
100 Epochs Batch size 128  Training Accuracy 86.8% but reaches this by epoch 25


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image7] 

The first 3 images turned out to be difficult to classify as they are all speed limits. These three are quite similar to each other. Image 9, the stay right sign, might have been difficult to identify as it was 'squished' by changing it to a 32x32x1 image. How ever it was identified correctly.


    

####2. My nine images with their correct label numbers are below:

    30kph label 1
    60kph label 3
    80kph label 5
    yield label 13
    no entry label 17
    dangerous curve to left label 19
    end restrictions label 32
    ahead only label 35
    keep right 38

The model was able to correctly guess 6 of the 9 traffic signs, which gives an accuracy of 66%. This could probably be improved by generating additional training images of all speed limit signs.

####3. The softmax probabliities are odd. Some of the cases like the first image that were completely wrong. My code predicted the wrong answer with a 92% probability of being right. Other guesses were wrong but with negligable confidence. Image 2 that had a confidence of only 19%.

Softmax makes the low performance more enlightening. The six signs it got right were all with fairly high degrees of certainty. The three it got wrong range from a certainty of 19% to 92%. Why? All three that it got wrong are speed limits. They all look similar to start with. They are all a big red circle with some numbers in the middle. Those numbers are taking up only a small percentage of the pixels. Ergo, a future version of this code should include a much larger sample set of speed limit signs.


The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

     image                                       prediction                  correct?
    30kph label 1                                   92%                        no
    60kph label 3                                   93%                        no
    80kph label 5                                   19%                        no
    yield label 13                                  99%                        yes
    no entry label 17                               99%                        yes
    dangerous curve to left label 19                97%                        yes
    end restrictions label 32                       88%                        yes
    ahead only label 35                             99%                        yes
    keep right 38                                   71%                        yes







Original Udacity READMD.md:

## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
