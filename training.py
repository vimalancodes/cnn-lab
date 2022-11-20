"""
This file is based on the TensorFlow tutorials
"""

"""
### Import TensorFlow
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from time import perf_counter


"""
Load our untrained CNN model 
"""
model = tf.keras.models.load_model('untrained-cnn-model')

"""
We load the training and testing data and normalize it. Normalizaing helps with numerical stability
"""
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

"""
Next, we define the training algorithm. In this case, we are using the ADAM algorithm. This is the most
commonly used one. 

We also define categotical cross-entropy as the loss function. This is the same
as the cross-entropy covered in class. However, we sum the probabilities over all categories. 

Finally, we define the performance metric. In this case, we are interested in how many images are being
classified correctly. Thus, we use accuracy.
"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
The next function performs the actual training. 
"""
tic = perf_counter()#start training timer
history = model.fit(train_images, 
                    train_labels, 
                    epochs=1, 
                    validation_data=(test_images, test_labels)
                    )
toc = perf_counter()# stop training timer
training_time = toc-tic #calculate total training time

"""
Finally, we save the training log, trining time, and the trained model for later analysis
"""
model.save('trained-cnn-model')

history_df = pd.DataFrame(history.history)
history_df.to_csv('cnn-training-history.csv')
np.savetxt('training_time.csv', [training_time], delimiter=',') 