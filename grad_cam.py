import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import keras
import keras.backend as K
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import load_model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from models import cnn_model
import sys
import cv2


def target_category_loss(x, category_index, nb_classes=10):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(path, img_shape=(224, 224)):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=img_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def compile_saliency_function(model, activation_layer='convolution2d_5'):
    input_img = model.input
    # layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    # layer_output = layer_dict[activation_layer].output
    last_conv_layer = filter(lambda l: "conv" in l.name,
                             reversed(model.layers))[0]
    layer_output = last_conv_layer.output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img], [saliency])


def modify_backprop(model, model_name, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        # new_model = VGG16(weights='imagenet')
        new_model = eval(model_name + '()')
    return new_model


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(input_model, image, category_index, layer_name,
             nb_classes=10):
    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: target_category_loss(x, category_index,
                                                  nb_classes)
    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers[0].layers
                   if l.name is layer_name][0].output
    # conv_output = filter(lambda l: "conv" in l.name,
    #                      reversed(model.layers[0].layers))[0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input],
                                   [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, image.shape[:2])
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def run_gradcam(model, model_name, image, true_label, layer_name):
    image = deprocess_image(image)
    preprocessed_input = np.expand_dims(image, axis=0)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    prob_predicted_class = np.max(predictions, axis=1)
    # top_1 = decode_predictions(predictions)[0][0]
    # print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
    print("True label {}, predicted label {}, with probability {}"
          .format(true_label,
                  predicted_class,
                  prob_predicted_class)
          )

    print('Predicted class: {}'.format(predicted_class))
    cam, heatmap = grad_cam(model, preprocessed_input,
                            predicted_class,
                            layer_name)
    cv2.imwrite("gradcam.jpg", cam)

    register_gradient()
    guided_model = modify_backprop(model, model_name, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
