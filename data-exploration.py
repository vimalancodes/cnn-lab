"""
# Convolutional Neural Network (CNN) Example
For details, go to https://www.tensorflow.org/tutorials/images/cnn
"""

"""
This script downloads the CIFAR dataset and displays some its images. 
[CIFAR images](https://www.cs.toronto.edu/~kriz/cifar.html). """

"""
### Import TensorFlow
"""
import tensorflow as tf
import matplotlib.pyplot as plt
"""

### Download and prepare the CIFAR10 dataset
The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each
class. The dataset is divided into 50,000 training images and 10,000 testing images. The
classes are mutually exclusive and there is no overlap between them.  """

# This function will download the data into the $HOME/.keras/datasets directory. 
# If the data already exists, it will not redownload. 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


"""
## We first create a plot that shows a few images from the trianing dataset. In more complex scenarions, 
you may run additional analysis of your dataset.  
"""
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.savefig('image_samples.png')