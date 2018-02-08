# **Traffic Sign Recognition** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[image1]: ./images/vis1.png "Sample Images"
[image2]: ./images/vis2.png "Histogram"
[image3]: ./images/vis3.png "New images"

Overview
---

Traffic sign recognition is has direct application in intelligent systems, including scene understanding and automated driving. In order to provide safety in automated driving, the vehicles must have the ability to identify various road signs so that it can adhere to the traffic rules imposed by them. 

The goal of this project is to build a system that identifies the category of the traffic sign. This is implemented via a convolutional neural network, trained and tested on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).



Dataset Summary 
---

The dataset is a collection of images from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Numpy functions were used to obtain basic statistics of the dataset.

* The size of training set is __34799__ images
* The size of the validation set is __4410__ images.
* The size of test set is __12630__ images.
* The shape of a traffic sign image is __(32, 32, 3)__.
* The number of unique classes/labels in the data set is __43__.


Exploratory Visualization
---

A few exploratory visualizations were done to get a better understanding of the dataset. First, images few categories of the dataset were plotted to verify that the categories and the images are consistent. The sample images plotted are from the classes: 8 (Speed limit(120km/h)), 14 (Stop), and 28 (Children crossing), respectively. 

Upon observing the images appears that each image shown are represented properly by their category labels. One point of interest, however, is that the images themselves can have (parts of) other signs in them. For example, the image of the speed limit sign on the top right has parts of another sign in the top half of the image. While these types of images are rare, too much of such images might affect training. 


![samples][image1]


In addition, histograms of the number of images per category for training, validation, and test sets, to get a sense of the frequency of each type of image in each set. It appears that the distribution of each category in each partition of the dataset are relatively the same. 

![hist][image2]

The list of categories and their corresponding class IDs are available at [here](https://github.com/enju244/CarND-Traffic-Sign-Classifier/blob/master/signnames.csv).



Data Preprocessing
---
The images are preprocessed prior to being fed by the network.

#### Grayscaling
The dataset images are originally in color (RGB), and they are converted to grayscale as a preprocessing step. Looking at the many categories, it appears that a lot of these signs share similar colors. For example, many categories of signs which have the colors red and white. Thus, color alone is not a good feature when it comes to classification of these signs.

On the other hand, the differentiaing features of the signs are their shapes and the letters and pictures drawn on them. By grayscaling, we remove color information from training, which would have been noisy information to the network.
The same may be valid for light exposure; among images of the same color, there are differences in the amount of light the sign is exposed to. This may also add noise to the training. 

The paper [Sermanet et al.](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) seems to suggest the use of the YUV color space. While this such color space conversion was also attempted, it did not lead to the best results, and thus were not used. It may be due possible errors in how I implemented the color space conversion.


#### Normalization
The image data were normalized with '(pixel - 128) / 128'. This way, the data will have zero mean and equal covariance. This scaling (theoretically) helps in neural networks by making training faster and reduce the chances of getting stuck in a local optima ([reference](https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network)).



Model Architecture
---

The model architecture was formed by an iterative process, with [LeNet-5](http://yann.lecun.com/exdb/lenet/) as the initial implementation. 

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Flatten			    | inputs 14x14x6, outputs 1176. Used later in FC layer |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten			    | inputs 5x5x16, outputs 400					|
| Concatenate		    | Combines previous flattend layers. inputs 1176 & 400, outputs 1576			|
| Fully connected		| inputs 1576, outputs 400						|
| RELU					| 												|
| Dropout				| keep_prob = 0.5 (for training only)			|
| Fully connected		| inputs 400, outputs 120						|
| RELU					| 												|
| Dropout				| keep_prob = 0.5 (for training only)			|
| Fully connected		| inputs 120, outputs 84						|
| Sigmoid				| 												|
| Dropout				| keep_prob = 0.5 (for training only)			|
| Fully connected		| inputs 84, outputs 43							|
| Softmax				| 	        									|
 

A list of modifications made from the original network are as follows:

- Addition of __Dropout__ layers in the fully connected layers.
- Use of __Xavier Initialization__, which provides better initialization for ConvNets. 
- Multi-Scale features: the flattened outputs of the first convolutional layer is also appended to the flattened output of the second convolutional layer before the fully connected layer. The idea was taken from [Sermanet et al.](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
- Additional fully connected layer: this provided better results, empirically. Note that the very last activation layer is Sigmoid instead of RELU. This is to avoid the values of the final layer to grow too large, which can saturate the softmax output.

The effectiveness of the changes are discussed in the "Model-building Approach and Testing" section.




Model Training
---

The following are the training hyperparameters used.

- Epochs: 20
- Batch Size: 128
- Learning Rate: 0.001
- Dropout Rate: 0.5
- Optimizer: Adam Optimizer 
- Loss operation: Reduce mean of softmax cross entrop



Model-building Approach and Testing
---

The first architecture tried was LeNet5, along with slight modifications with the input and output dimensions. However, this was producing an accuracy ofaround 0.85 for validation set in 10 epochs. 

As discussed in the Data Preprocessing section, I thought that color may have been an issue. Due to this, grayscale was implemented, raising the validation accuracy up to around 0.89 in 10 epochs. 

The next idea was to add dropout. I thought this may halp in a few ways. First, it would set up the network as an ensemble classifier, leading to stronger classification. Second, it would help avoid overfitting by acting as a form of regularization. The addition of dropout layers improved the validation accuracy. Since the validation accuracy seemed to be improving after 10 epochs, the number of epochs was adjusted to 20. 

A few more ideas were added and kept based on empirical results. Each one helped increase the validation accuracy slightly, helping provide a final validation accuracy over the required 0.93. 

Xavier initialization was used in place of the tf.truncated_normal initialization. This method of initialization helps make sure that the weights are 'just right', "keeping the signal in a reasonable range of values throughout many layers" ([reference](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)).

Multi-scale feature was a configuration taken from [Sermanet et al.](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Instead of using just the flattened output of the second convolutional layer as the input to the fully connected layer, the output of the first layer is also used. The intuition here is that "higher-resolution" features from the first conv. layer is also used directly for classification, "providing different scales of receptive fields to the classifier."

Not much parameter tuning were done extensively; however, searching for the best set of hyperparameters in the parameterspace may potentially provide better prediction accuracies.


My final model results were as follows:
- training set accuracy of __0.995__
- validation set accuracy of __0.956__
- test set accuracy of __0.934__




Testing on New Images
---

Here are five German traffic signs that I found on the web:

![webimgs][image3]

Out of these, the second image may be the most difficult to classify. Note that this image not only includes the sign for 'No passing for vehicles over 3.5 metric tons', but another sign below that denotes the name of the bridge. As mentioned in the Exploratory Visualizations Section, such images may be difficult for the model to predict on.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animal crossing	| Wild animal crossing			 				|
| 3.5 metric tons     	| Slippery road 								|
| Ahead only			| Ahead only 	    							|
| Road work				| Road work										|
| 30 km/h 	    		| 30 km/h    									| 

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.3%, given that this small set only contained 5 images, and that the second image was deemed to be difficult to classify. 



Testing on New Images, Top-K prediction
---

The Top-5 predictions softmax probabilities are also calculated for each of the new test images. For all five of the images, the classifier is confident about it's prediction (in which, one of the five, is unfortunately incorrect). 

For the first, third, fourth, and fifth images, the model is very sure of its prediction, and it's prediction is correct.

First image: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.98311877e-01	  	| Wild Animals Crossing 						| 
| 8.70956050e-04		| Double curve									|
| 6.64652849e-04		| Dangerous curve to the left					|
| 3.40241386e-05		| Speed limit (80km/h)							|
| 3.21377993e-05    	| Keep left										|


Third image: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99887109e-01		| Ahead only									| 
| 3.82104372e-05		| Speed limit (60km/h)							|
| 2.98794894e-05		| Yield											|
| 1.84326218e-05		| Turn left ahead								|
| 6.72972055e-06		| Turn right Ahead								|

Fourth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99814570e-01		| Road work				 						| 
| 6.32341907e-05		| Dangerous curve to the right					|
| 5.71494566e-05		| Bumpy road									|
| 2.80476670e-05		| Bicycles crossing								|
| 1.99627721e-05 		| Beware of ice/snow							|
            

Fifth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.98820841e-01		| Speed limit (30km/h)							|
| 7.27134466e-04		| Speed limit (50km/h)							|
| 2.18735207e-04 		| Speed limit (20km/h)							|
| 2.14975284e-04   		| Speed limit (70km/h)							|
| 5.48432172e-06		| Go straight or left							|


     
The result on the second image is interesting. While it is again somewhat confident about it's top-1 prediction, none of its top-5 guesses are actually correct. Relative to the results for the rest of the images above, it's evident that it is not as confident about the prediction. Again, this was somewhat expected as the second image was the one that was predicted to be difficult to predict due to the additional second sign within the image which can confuse the network.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.77352715			| Slippery road 								| 
| 0.11145856    	 	| Dangerous curve to the right 					|
| 0.04131101 			| Dangerous curve to the left					|
| 0.02406631			| Speed limit (60km/h)							|
| 0.02145814		    | Ahead only 			     					|

