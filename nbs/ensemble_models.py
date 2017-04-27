
# coding: utf-8

# In[18]:

from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input
from keras.utils import np_utils
from keras.datasets import cifar10
# import matplotlib
# matplotlib.use('Agg')


# Test the hypothesis of pretrained model on imagenet against adversarial perturbations

# In[5]:

model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(32, 32, 3)))


# In[6]:

# check the model output dimension
model.output


# In[7]:

# check weather we can take the conv feature map
batch_size = 128
nb_classes= 10
nb_epoch = 100 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
trY = np_utils.to_categorical(y_train)
teY = np_utils.to_categorical(y_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.


# In[8]:

# conv feature maps
trX = model.predict(X_train)
teX = model.predict(X_test)


# In[12]:

print trX.shape
print teX.shape
model.output


# In[30]:

# construct thte top layer
model2 = Sequential([
    Flatten(input_shape=trX.shape[1:]),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(10, activation='softmax')
])

model2.summary()


# In[ ]:

model2.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(trX, trY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(teX, teY), shuffle=True)


# In[ ]:

model2.evaluate(teX, teY)
model2.save('vgg16_model_tained_top.h5')
model2.save_weights('vgg16_weights_trained_top.h5')

