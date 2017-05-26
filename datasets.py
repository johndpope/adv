'''Import the dataset containing the training images for the convolutional
neural network
'''

import numpy as np
import scipy.ndimage
import scipy


def load_images_torcs_4():
    import glob
    #filepath = 'datasets/torcs_4/training_set/*.png'
    filepath = '../../voc2012/subset/*.jpg'
    filenames = glob.glob(filepath)

    sample = scipy.misc.imresize(scipy.ndimage.imread(filenames[0]), (64, 64))
    num_images = len(filenames)
    images = np.zeros((num_images, sample.shape[0], sample.shape[1], 1), dtype=np.uint8)

    for i in range(num_images):
        images[i] = scipy.misc.imresize(scipy.ndimage.imread(filenames[i]), (64, 64))[:, :, :1]

    images = images.reshape(len(images), 64, 64, 1)
    #images = images.reshape(len(images), 1, 64, 64)
    print("images = {}".format(images.shape))

    return images
