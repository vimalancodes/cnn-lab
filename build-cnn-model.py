"""
### Import TensorFlow
"""
import tensorflow as tf

"""
### Create the convolutional neural network model. 
"""

"""
The 6 lines of code below define the convolutional base. That is, the convolution layers of our CNN.
As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring
the batch size. If you are new to these dimensions, color_channels refers to (Red,Green,Blue). In this
example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the
format of CIFAR images (i.e., 32x32 pixel colored images). You can do this by passing the
argument `input_shape` to your first layer.
"""

""" By using the Sequential() class, this code builds the convolutional network layer by layer  """
model =   tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))


"""
You can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor (i.e., three 2D
matrices stacked on top of each other) of
shape (height, width, channels). The width and height dimensions tend to shrink as you go
deeper in the network. The number of output channels for each Conv2D layer is controlled by the
first argument (e.g., 32 or 64). Typically,  as the width and height shrink, you can afford
(computationally) to add more output channels in each Conv2D layer.  """
"""

## Add Dense layers on top
The above model defines a convolutional neural network with two convolutional layers (convolution
operaion+pooling operation).
To complete the model, you will feed the last output tensor from the convolutional neural network (of
shape (4, 4, 64)) into one or more Dense layers to perform classification. Dense layers take
vectors as input (which are 1D), while the current output is a 3D tensor. First, you will
flatten (or unroll) the 3D output to 1D,  then add one or more Dense layers on top. CIFAR has
10 output classes, so you use a final Dense layer with 10 outputs.  """

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10)) 

"""
The final output layer with 10 neurons. Each neuron gives the a
probability of how likely the input image is to belong to the class that corresponds to the
neuron. 
"""

"""
We can verify that we are building the model we want by displaying a summary of the layers 
"""
model.summary()

"""
We can also verify the model by observing a diagram
Pydot can be installed by running "pip install pydot"
"""
tf.keras.utils.plot_model(model, to_file="cnn_model.png", show_shapes=True)

"""
We save the model to train it later
"""
model.save('untrained-cnn-model')

