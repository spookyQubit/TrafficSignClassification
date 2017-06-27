# Traffic Sign Recognition 

---

### Introduction

The goal of this project is to build a traffic sign recognition model which can classify images from the [German Traffic Sign Benchmarks](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The learning model is based on convolutional neural networks. The architechture used is a modification of the original [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) setup.

Results
* Training set accuracy of 1.00%
* Validation set accuracy of 97.2%
* Test set accuracy of 95.2%

[//]: # (Image References)

[image1]: ./examples/training_images_count.jpg "Training data class counts"
[image2]: ./examples/training_valid_images_count.jpg "Training test data ratio"
[image3]: ./examples/original_images.png "Original images"
[image4]: ./examples/augmented_image.png "Augmented image"
[image5]: ./examples/transformed.png "Transformed image"
[image6]: ./examples/confusion_matrix.png "Confusion matrix"
[image7]: ./examples/new_images.png "New images"
[image8]: ./examples/placeholder.png "Traffic Sign 4"
[image9]: ./examples/placeholder.png "Traffic Sign 5"

---

Detailed implementation of the model can be found in the link to my [project code](https://github.com/spookyQubit/TrafficSignClassification/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### Basic summary of the data set. 

* The size of training set is 34799 (this was increased to 34799 x 2 after augmentation)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43


We first look at the reletive distribution of the various classes in the training dataset. 
![alt text][image1]
From the above plot, it is clear that the training data is not uniformaly distributed amongst the 43 different classes. In particular, we have 11 times more data points for "Speed limit (50km/h)" as compared to "Speed limit (20km/h)".

Next we check if the distribution of classes in the validation set is similar to that in the training data.  
![alt text][image2]
The plot suggests that the distribution of the number of labes in training and validation datasets are different.

#### Input Images
A peek into the images in the training dataset shows us that some of the images are difficult to classify. For example, notice the "No passing" image below. 

![alt text][image3]

#### Data Augmentation
For each training image, an augmented image was generated. The following transformations were done for augmentation:

```Augmentation
* random_brightness: Randomly change the brigntness of the image
* random_rotation: Randomly changing the angle of the image. In order to prevent loss of information contained in angles, the random angles with with the images were rotated were picked from a distribution with a small standard deviation.
* random_translation: Randomly move the image
* random_shear: Randomly distort the image
```
The fact that each training image had a corresponding augmented image ensured that the relative distribution of the class labels did not change after agugmentation. Increasing the training datasize did increase the time it took to train the model but it helped to reduce overfitting. 

An example of an original image with its corresponding agmented image is shown below.

![alt text][image4]

### Preprocessing
Each image in the training/validation/test set was scaled to have zero mean and unit variance.   
![alt text][image5]

### Model Architecture
The model was based on the LeNet architechture. The following table shows the dimentionality and the type of the layers used in the model. 

Layer # | Layer Type | Output Shape
:---:| :--- | :---
**Input**||32x32x3 (3 because there are three color channels)
**1**| Convolutional | 28x28x24
**1** | Activation (Relu) | 28x28x24
**1**| Pooling (Max) | 14x14x24
**2**| Convolutional | 10x10x64
**2** | Activation (Relu) | 10x10x64
**2**| Pooling (Max) | 5x5x64
| Flatten | 1600
**3**| Fully Connected | 480
**3**| Activation (Relu) | 480
**4**| Fully Connected | 43 (The number of unique labes in the training data)
**Output**| Logits | 43


#### Training the model

To train the model, I used tensorflow's **AdamOptimizer**. The various hyperparameters which were considered while training were:
1) Number of layers
2) Number of feature maps in convolutional layers
3) Number of neurons in fully connected layers
4) Droupout probability
4) Number of epochs
6) Learning rate
7) Batch size  


An iterative approach was chosen to come up with the final model:
* Initial architecture: At first, a LeNet architechture with five layers were chosen with parameters same as in LeNet5 with {learning_rate: 0.001, BATCH_SIZE: 100, EPOCH: 5}. No dropout was used. 
* Problems with the first architecture: The initial architechture had a huge gap between training and validation accuracy. Also, the validation acuracy was around 91%, not acceptable as per the requirements of the project.  
* Adjustment of architecture: In order to increase the accuracy, I increased the number of feature maps in the convolutional layers. For first layer, feature maps was increased from 6 to 12. For the second convolutional layer, the number of feature maps were increased from 16 to 32. 
This adjustment increased the accuracy but still there was considerable overfitting. In order to address the problem, dropout was added to the fully connected layer. Also one of the fully connected layer was completeley dropped. This considerably reduced overfitting and gave a validation accuracy of around 96%. 
Once this architechture was decided, the learning rate was lowered the batch size was reduced and the number of epochs were increased to 50. 
* The initial weights of the layers were chosen as discussed by [Andrej Karapathy](http://cs231n.github.io/neural-networks-2/)

For the final model, the confusion matrix evaluated on the validation dataset is

![alt text][image6]


Final results
* Training set accuracy of 1.00%
* Validation set accuracy of 97.2%
* Test set accuracy of 95.2%


### Test a Model on New Images

The model's ability to predict on new traffic signs was tested on 5 new images. These images were rescaled/resized to 32x32x3 numpy arrays and the same pre-processing step was applied to them before feeding it to the model. The images and the top five classes predicted by the model with their corresponding probabilites is shown below:

![alt text][image7]

The model was able to correctly classify 4 out of the five new images thus having an accuracy of 80%. The one image which was incorrectly classified is the Speed limit (80km/h) sign which was predicted as Speed limit (30km/h). The reason of this could be because of the incorrect featurization of the digit 8 which may have led to 8~3. 

### Conclusion and possible improvements

1) It would have been very helpful to take advantage of sklearn's pipeline feature. This would have made the code cleaner and made iterations in choosing model architecture less time consuming.
2) I did not use GPU as I found that the AMI on AWS did not have a couple of packages (like cv2) which I was considering to use.
3) There are some classes which have low accuracy, for example "Roundabout mandatory". It might help to add more data and tweak the pre-processing pipeline to increase the accuracy of these classes. 
4) It will be interesting to see the performace by choosing a completely different architecture, for example AlexNet. 

### References:
1) A great read for traffic sign classification by [Alex Staravoitau](https://navoshta.com/traffic-signs-classification/)
2) Augmentation idea from inspired from [naokishibuya](https://github.com/naokishibuya/car-traffic-sign-classification)
