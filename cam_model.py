import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv2D, Dropout, Flatten, Dense, Input
# from vis.utils import utils
# from vis.visualization import visualize_cam, visualize_activation, get_num_filters
# from scipy.misc import imsave, imread, imresize
from config import ga_setup
from utils import ga_train
from models import resnet
from keras.utils.vis_utils import plot_model


def create_model(data_shape=(28, 28, 1)):
    inpt = Input(shape=data_shape)
    # x = Dropout(0.2, input_shape=data_shape)(inpt)
    x = Conv2D(64, (8, 8), strides=(2, 2),
               padding='same', activation='relu')(inpt)
    x = Conv2D((128), (6, 6), strides=(2, 2),
               padding='valid', activation='relu')(x)
    x = Conv2D((128), (5, 5), strides=(1, 1),
               padding='valid', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inpt, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # K.set_learning_phase(0)
    tf.set_random_seed(2017)
    sess = tf.Session()
    K.set_session(sess)
    # annot = './voc2012/Annotations/'
    # jpegs = './voc2012/JPEGImages/'
    (trX, trY), (teX, teY) = mnist.load_data()
    # trX = np.load('./voc2012/trX_3000_samples.npy')
    trX = trX / 255.
    trX = np.expand_dims(trX, axis=3)
    teX = teX / 255.
    trY = np_utils.to_categorical(trY, 10)
    teY = np_utils.to_categorical(teY, 10)
    # trY = np.load('./voc2012/trY.npy')
    model = create_model()
    # model, name = resnet()
    # model = load_model('mnist_model.hdf5')
    model.summary()
    plot_model(model, to_file='cnn_model.png', show_shapes=True,
               show_layer_names=True)
    import pdb
    pdb.set_trace()
    toolbox = ga_setup(model, trX)
    ga_train(model, trX, toolbox)
    # model.fit(trX, trY, validation_split=0.21, batch_size=128, epochs=2,
    #           verbose=1)
    # model.save('mnist_model.hdf5')
    # plot original image
    # from filter_vis import vis_filter
    # vis_filter(model, 'conv2d_3', img_width=224, img_height=224,
    #            nb_filters=100, nb_iter=100)
    import pdb
    pdb.set_trace()
    img_idx = np.random.randint(0, len(trX))
    img = trX[img_idx]
    img = imresize(img, (224, 224))
    # img = imread('mnist_img.jpg')
    plt.imshow(img)
    plt.title('Original Image')
    plt.show()
    # img = np.expand_dims(img, axis=0)
    # print("image shape = {}".format(img.shape))
    # print("img = {}".format(img.shape))
    # imsave('mnist_img.jpg', img)

    # predicted and true class
    # pred_class = np.argmax(model.predict(img))
    img = np.expand_dims(img, axis=0)
    pred_class = np.argmax(model.predict(img))
    # actual_class = np.argmax(trY[img_idx])
    print("predicted class = {}".format(pred_class))
    # import pdb
    # pdb.set_trace()
    # layer_name = 'block5_conv3'
    # layer_idx = [idx for idx, layer in enumerate(model.layers)
    #              if layer.name == layer_name][0]
    # run_gradcam(model, 'cnn_model', image,
    #             pred_class, layer_name)
    # layer_idx = 6
    # heatmap = visualize_cam(model, layer_idx, [pred_class], img,
    #                         penultimate_layer_idx=2)
    # print("heatmap = {}".format(heatmap.shape))
    # plt.axis('off')
    # # plt.imshow(heatmap.reshape(28, 28, 3))
    # plt.imshow(heatmap.reshape(224, 224, 3))
    # plt.title('Saliency map')
    # plt.show()
    # # dense layer visualization
    # # generate 3 different images of the same output index
    # cmap = plt.get_cmap('jet')
    # vis_images = []
    # for idx in [7, 8, 9]:
    #     tmp_img = visualize_activation(model, layer_idx, filter_indices=idx,
    #                                    max_iter=500)
    #     tmp_img = utils.draw_text(tmp_img, str(idx))
    #     vis_images.append(tmp_img)

    # stiched = utils.stitch_images(vis_images)
    # plt.axis('off')
    # plt.imshow(stiched)
    # plt.title('predictions')
    # plt.show()
    # conv layer visualization
    layer_idx = 3
    # visualize all filters in this layer
    filters = np.arange(get_num_filters(model.layers[layer_idx]))
    # Generate input image for each filter. Here 'text' field is used
    # to overlay 'filter_value' on top of the image
    vis_images = []
    for idx in filters:
        tmp_img = visualize_activation(model, layer_idx, filter_indices=idx)
        # tmp_img = utils.draw_text(tmp_img, str(idx))
        vis_images.append(tmp_img)

    # Generate stiched image palette with 8 colors
    stitched = utils.stitch_images(vis_images, cols=8)
    plt.axis('off')
    plt.imshow(stitched)
    plt.title('conv2d_2')
    plt.show()
    # heatmap2 = gradcam(model, img, [pred_class],
    #                    layer_idx, 2)
    # print("heatmap2 = {}".format(heatmap2.shape))
    # plt.axis('off')
    # plt.imshow(heatmap.reshape(28, 28, 3))
    # plt.title('Saliency map')
    # plt.show()
    # sess.close()
