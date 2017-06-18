import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import keras
import keras.backend as K
# from keras.models import load_model
from keras.layers.core import Lambda
from keras.models import Sequential
# import sys
import cv2
# from models import cnn_model


def target_category_loss(x, category_index, nb_classes=10):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def prepare_image(x):
    # img_path = sys.argv[1]
    # img = image.load_img(path, target_size=img_shape)
    # x = image.img_to_array(img)
    # x = np.float32(np.expand_dims(x, axis=0))
    # x = preprocess_image(x)
    x = deprocess_image(x)
    print("image shape: {}\n image max pixel val: {}".format(x.shape,
                                                             np.max(x)))

    return x


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def compile_saliency_function(model, activation_layer='conv2d_2'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    # last_conv_layer = filter(lambda l: "conv" in l.name,
    #                          reversed(model.layers))[0]
    # layer_output = last_conv_layer.output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


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
        # new_model = eval(model_name +
        #   '(img_rows=224, img_cols=224, channels=3, nb_classes=20)')
        new_model = model
        # new_model.summary()
    return new_model


def preprocess_image(x):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68

    return x


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


def grad_cam(input_model, img, category_index, layer_name,
             nb_classes=10):

    model = Sequential()
    model.add(input_model)
    target_layer = lambda x: target_category_loss(x,
                                                  category_index,
                                                  nb_classes)
    model.add(Lambda(target_layer,
              output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers[0].layers
                   if l.name == layer_name][0].output
    # conv_output = filter(lambda l: "conv" in l.name,
    #                      reversed(model.layers[0].layers))[0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input,
                                    K.learning_phase()],
                                   [conv_output, grads])

    output, grads_val = gradient_function([np.expand_dims(img, axis=0), 0])
    # grads_val = np.nan_to_num(grads_val)
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # ReLU
    cam = np.maximum(cam, 0)
    # cam = cam / (np.max(cam) + 1e-5)
    heatmap = cam / (np.max(cam) + 1e-5)
    # cam = cv2.resize(cam, tuple(img.shape[:2][::-1]))
    cam = cv2.resize(cam, img.shape[:2])

    # Return to BGR [0..255] from the preprocessed image
    # imge = img[0, :]  # this is analogous to np.squeeze()
    # imge -= np.min(img)
    # imge = np.minimum(img, 255)

    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # cam = 255.0 * (np.float32(cam) + np.float32(img))
    cam = 1.0 * np.float32(cam) + np.float32(img)
    cam = cam / np.float32(np.max(cam) + 1e-5)

    # return np.uint8(cam), heatmap
    return cam, heatmap


def run_gradcam(model, model_name, img, class_label, layer_name):
    # image = prepare_image(image)
    predictions = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(predictions)
    prob_predicted_class = np.max(predictions, axis=1)
    print("True label {}, predicted label {}, with probability {}"
          .format(class_label,
                  predicted_class,
                  prob_predicted_class)
          )

    print("image shape {}".format(img.shape))
    cam, heatmap = grad_cam(model, img, predicted_class, layer_name,
                            nb_classes=10)
    print("cam: {}, heatmap: {}".format(cam.shape, heatmap.shape))
    cv2.imwrite("gradcam.jpg", cam)

    return cam, heatmap

    # register_gradient()
    # guided_model = modify_backprop(model, model_name, 'GuidedBackProp')
    # saliency_fn = compile_saliency_function(guided_model, layer_name)
    # saliency = saliency_fn([image, 0])
    # gradcam = saliency[0] * heatmap[..., np.newaxis]
    # cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(cam)
    # axes[1].imshow(np.squeeze(gradcam))
    # plt.show()
