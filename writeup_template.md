#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/grayscale2.jpg "Grayscaling After"
[image4]: ./examples/random_noise.jpg "Random Noise"
[image5]: ./examples/placeholder.png "Traffic Sign 1"
[image6]: ./examples/placeholder.png "Traffic Sign 1"
[image7]: ./examples/placeholder.png "Traffic Sign 2"
[image8]: ./examples/placeholder.png "Traffic Sign 3"
[image9]: ./examples/placeholder.png "Traffic Sign 4"
[image10]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Andrew-Hogan/Traffic-Sign-Neural-Network/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

The first visualization is simply a plot of the original data without any processing. The second visualization, found in the fourth cell, shows the distribution of the data over each class. There are clear differences in the frequency of each class; however, there are also clear differences in the frequency of signs seen while driving.

![Histogram][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because leading literature shows that while converting to grayscale may reduce accuracy at the highest levels of precision due to loss of data, it allows the neural network to train faster by ignoring data that is not necessary. I also zero centered the data for similar reasons.

Here is an example of a traffic sign image before and after grayscaling.

![before][image2]
![after][image3]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the 11th code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the sklearn library's train_test_split function. 

My final training set had 188203 number of images. My validation set and test set had 47051 and 12630 number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because there were relatively few samples of many different classes of traffic signs. I also decided to augment the data before splitting into validation data. The result was a very robust neural network, which initially performed better on testing data than it did on validation or training data - and continued to have minimal differences between the three for each epoch once I had finalized my network architecture. It forced the neural network to only rely upon the abstract concepts within each sign rather than any positional correlations. To add more data to the the data set, I used translation, roatation, and shearing as these transformations do not modify the abstract concepts within each sign.

Here is an example of an augmented "no parking" sign:

![augmented][image4]

I did this five times per image, resulting in 188203 training samples and 47051 validation samples.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 12th cell of the ipython notebook. 

My final model consisted of the following layers:

Input = 32x32x1 Grayscale Image
Layer 1 = 3x3x1x12 Convolutional Layer with valid padding and 1 stride. Outputs 30x30x12 with a Relu activation.
Layer 2 = 5x5x12x24 Convolutional Layer with valid padding and 1 stride. Outputs 26x26x24 with a Relu activation.
A Max Pooling layer (valid padding) with an output of 13x13x24.
Layer 3 = 6x6x24x32 Convolutional Layer with valid padding and 1 stride. Outputs 8x8x32 with a Relu activation.
A Max Pooling layer (valid padding) with an output of 4x4x32.
A Flattened layer which outputs with 512 neurons.
Layer 4 = 512x172 Fully Connected Layer which outputs 172, with a ReLu activation.
Layer 5 = 172x86 Fully Connected Layer which outputs 86, with a ReLu activation.
Layer 6 = 86x43 Fully Connected Layer which outputs the 43 logits. 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 13th cell of the ipython notebook. 

To train the model, I used a batch size of 128, 50 epochs, and a .001 learning rate with an Adam Optimizer. However, I saw no real gain in training or testing accuracy after 20 epochs and could have stopped there. I would have included dropout, and in fact the code is still there (with a keep_prob of 1) - but the model I used was not wide or deep enough to allow for any significant dropout implementation without hurting performance. I would have created a larger architecture, but the performance was acceptable and any changes would have been at a significant cost in terms of time.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 13th and 14th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 97.3% 
* test set accuracy of 96.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The original architecture which was chosen was the LeNet-5 architecture from the classroom. This was a logical and easy starting point, as it performed well at 88% test accuracy at a very quick rate.
* What were some problems with the initial architecture?
It suffered from an inability to correctly abstract the different concepts within the set of traffic signs. It was not deep enough and yet it was still over-fitting to the training data without any augmentation or preprocessing.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Augmentation was adjusted to allow the network to be made deeper without overfitting. However, upon adding multiple layers, inception layers, and feedforward layers, the network had thousands of neurons in the first fully connect layer. Even worse, it was now performing worse than the original model as the data augmentation combined with the plethora of inputs made it difficult for the model to create any meaningful changes in weights using gradient descent. I got the impression that the model was learning some abstractions but having difficulty applying them to all the augmented data, as the validation set quickly converged to about 75% accuracy while the test set accuracy was at around 85%. This underfitting paradox was confusing at first, but I decided to simplify the model with more meaningful layers and specific goals in mind for each layer. I changed the first layer, the 5x5, into a 3x3 to look for smaller patterns within the data and avoided maxpooling to allow the model to look for more nuanced patterns within that data in the next layer. Afterwords, I reduced the size for training time's sake through max pooling and again looked for larger patterns with a 6x6. Finally, for each fully connected layer, I had it correlated to the size of the final outputs (43) in what may have been a naive attempt at having each output correlated with a simplified set of neurons in each fully connected layer with extras in case there would be dead neurons - as I used a ReLu activation instead of a Leaky ReLu activation.
* Which parameters were tuned? How were they adjusted and why?
The learning rate was adjusted to allow it to converge while still using gradient descent. I would have liked to use a deeper network, but this trained the fastest given the multitude I had tried, and tuning after training networks with thousands of neurons in a fully connected layer with only 43 possible outputs seemed wasteful.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![first][image5] ![second][image6] ![third][image7] 
![fourth][image8] ![fifth][image9]

The rough road image may be difficult to classify, as it looks very similar to many other traffic signs and contains a very small amount of information to clue the network in to which type of caution sign it is.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      		| General Caution   									| 
| No Passing     			| No Passing 										|
| Road Work					| Right of Way at Next Intersection											|
| Bumpy Road	      		| Bumpy Road					 				|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is typical given the ~3% missing accuracy in the test set. The miscalculated road work sign does admittedly look very similar in terms of data distribution to the priority at the next intersection sign. This problem would be present for any upward facing triangle sign with a manlike image in the center.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a General Caution sign (probability of 1.0e+00), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General Caution   									| 
| 3.7912e-23     				| Traffic Signals 										|
| 9.77026e-25					| Pedestrians											|
| 1.655e-30	      			| Road Work					 				|
| 2.7844e-35				    | Slippery Road      							|


For the second image, the model is very sure that this is a No Passing sign (probability of 1.0e+00), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0e+00         			| No Passing   									| 
| 6.17061213e-29     				| Vehicles over 3.5 metric tons prohibited 										|
| 0.00e+00					| Speed limit (20km/h)											|
| 0.00e+00	      			| Speed limit (30km/h					 				|
| 0.00+00				    | Speed limit (50km/h      							|

For the third image, the model is less sure (but still certain) that this is a Right-of-way at the next intersection
 sign (probability of 1.0e+00), but the image contains a Road Work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0e+00         			| Right-of-way at the next intersection   									| 
| 3.08112513e-09     				| Traffic signals 										|
| 1.89050386e-10					| Pedestrians											|
| 2.51815474e-12	      			| General caution					 				|
| 1.04265717e-12				    | Ahead only      							|

For the fourth image, the model is very sure that this is a bumpy road sign (probability of 1.0e+00), and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0e+00         			| Bumpy road   									| 
| 1.64725949e-34     				| Slippery road 										|
| 0.00e+00					| Speed limit (20km/h)											|
| 0.00e+00	      			| Speed limit (30km/h					 				|
| 0.00+00				    | Speed limit (50km/h      							|

For the fifth image, the model is very sure that this is a Stop sign (probability of 1.0e+00), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0e+00         			| Stop   									| 
| 3.76357097e-20     				| Speed limit (30km/h) 										|
| 1.21002468e-20					| Turn right ahead											|
| 1.56813431e-22	      			| Traffic signals					 				|
| 1.52387631e-22				    | Keep right      							|
