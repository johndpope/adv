from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
# from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras.layers import MaxPooling2D, Input, Lambda, Conv2D, UpSampling2D
from keras.layers import LSTM, TimeDistributed, merge, SimpleRNN, ELU
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras import initializers
# from keras.initializations import normal, identity
# from keras.initializers import normal
from keras import backend as K
from square_layer import SquareMulLayer
from vae_loss import CustomVariationalLayer

np.random.seed(2017)  # for reproducibility


def set_img_ordering(data_shape):
    if K.image_dim_ordering() == 'th':
        input_shape = data_shape
    else:
        input_shape = data_shape

    return input_shape


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) *
                  K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() > 0.5))


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def create_siamese_data(X_train, y_train, X_test, y_test):
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(X_test, digit_indices)

    return tr_pairs, tr_y, te_pairs, te_y


def siamese(data_shape):
    # network definition
    base_network = create_base_network(data_shape)

    input_a = Input(shape=(data_shape,))
    input_b = Input(shape=(data_shape,))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)
    ([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)

    return model

    # compute final accuracy on training and test sets
    # pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    # tr_acc = compute_accuracy(pred, tr_y)
    # pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    # te_acc = compute_accuracy(pred, te_y)


def convresblock(x, nfeats=8, ksize=3, nskipped=2, elu=True):
    ''' The proposed residual block from [4]'''
    y0 = Conv2D(nfeats, ksize, ksize, padding='same')(x)
    y = y0
    for i in range(nskipped):
        if elu:
            y = ELU()(y)
        else:
            y = BatchNormalization(axis=1)(y)
            y = Activation('relu')(y)
            y = Conv2D(nfeats, (ksize, ksize), padding='same')(y)
    return merge([y0, y], mode='sum')
    # return layers.add([y0, y])


def getwhere(x):
    ''' Calculate the "where" mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x
    return K.gradients(K.sum(y_postpool), y_prepool)


def preprocess_swwae(X_train, X_test):
    if K.backend() == 'tensorflow':
        raise RuntimeError('This example can only run with the '
                           'Theano backend for the time being, '
                           'because it requires taking the gradient '
                           'of a gradient, which isn\'t '
                           'supported for all TF ops.')
    K.set_image_data_format('channels_first')

    # The size of the kernel used for the MaxPooling2D
    pool_size = 2
    # The total number of feature maps at each layer
    nfeats = [8, 16, 32, 64, 128]
    # The sizes of the pooling kernel at each layer
    pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
    # The convolution kernel size
    ksize = 3

    if pool_size == 2:
        # if using a 5 layer net of pool_size = 2
        X_train = np.pad(X_train, [[0, 0], [0, 0], [2, 2], [2, 2]],
                         mode='constant')
        X_test = np.pad(X_test, [[0, 0], [0, 0], [2, 2], [2, 2]],
                        mode='constant')
        nlayers = 5
    elif pool_size == 3:
        # if using a 3 layer net of pool_size = 3
        X_train = X_train[:, :, :-1, :-1]
        X_test = X_test[:, :, :-1, :-1]
        nlayers = 3
    else:
        import sys
        sys.exit("Script supports pool_size of 2 and 3.")

    # Shape of input to train on (note that model is fully
    # convolutional however)
    input_shape = X_train.shape[1:]

    return input_shape, nfeats, nlayers, pool_sizes, ksize


def swwae(X_train, X_test):
    input_shape, nfeats, nlayers, pool_sizes, ksize = preprocess_swwae(X_train,
                                                                       X_test)
    # The final list of the size of axis=1 for all layers, including input
    nfeats_all = [input_shape[0]] + nfeats

    # First build the encoder, all the while keeping track of the "where" masks
    img_input = Input(shape=input_shape)

    # We push the "where" masks to the following list
    wheres = [None] * nlayers
    y = img_input
    for i in range(nlayers):
        y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
        y = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)
        wheres[i] = merge([y_prepool, y], mode=getwhere,
                          output_shape=lambda x: x[0])

    # Now build the decoder, and use the stored "where" masks to places
    # the features
    for i in xrange(nlayers):
        ind = nlayers - 1 - i
        y = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
        y = merge([y, wheres[ind]], mode='mul')
        # y = layers.multiply([y, wheres[ind]])
        y = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

    # Use hard_simgoid to clip range of reconstruction
    y = Activation('hard_sigmoid')(y)

    # Define the model and it's mean square error loss, and
    # compile it with Adam
    model = Model(img_input, y)
    model.compile('adam', 'mse')

    return model
    # Plot
    # X_recon = model.predict(X_test[:25])
    # X_plot = np.concatenate((X_test[:25], X_recon), axis=1)
    # X_plot = X_plot.reshape((5, 10, input_shape[-2], input_shape[-1]))
    # X_plot = np.vstack([np.hstack(x) for x in X_plot])
    # plt.figure()
    # plt.axis('off')
    # plt.title('Test Samples: Originals/Reconstructions')
    # plt.imshow(X_plot, interpolation='none', cmap='gray')
    # plt.savefig('reconstructions.png')


def hierarchical(data_shape, nb_classes=10,
                 row_hidden=256, col_hidden=256):
    # Embedding dimensions.
    # 4D input.
    x = Input(shape=data_shape)

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(units=row_hidden))(x)

    # Encodes columns of encoded rows.
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    reshaped = Reshape((16, 16))(encoded_columns)
    encoded_cols = LSTM(256)(reshaped)

    # Final predictions and model.
    prediction = Dense(nb_classes, activation='softmax')(encoded_cols)
    model = Model(inputs=x, outputs=prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def irnn(data_shape, nb_classes=10,
         learning_rate=1e-6, hidden_units=784):
    # clip_norm = 1.0
    model = Sequential()
    model.add(SimpleRNN(hidden_units,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu', input_shape=data_shape))
    model.add(Reshape((28, 28)))
    model.add(SimpleRNN(625,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu'))
    model.add(Reshape((25, 25)))
    model.add(SimpleRNN(256,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu'))
    model.add(Reshape((16, 16)))
    model.add(SimpleRNN(100,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def mlp(data_shape):
    model = Sequential()
    model.add(Dropout(0.5, input_shape=data_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(784))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def cnn_model(data_shape, logits=False, input_ph=None, nb_filters=64,
              nb_classes=10):
    """
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """

    model = Sequential([
        Dropout(0.5, input_shape=data_shape),
        Conv2D(nb_filters, (8, 8), strides=(2, 2),
               padding='same', activation='relu',
               input_shape=data_shape),
        Conv2D((nb_filters * 2), (6, 6), strides=(2, 2),
               padding='valid', activation='relu'),
        Conv2D((nb_filters * 2), (5, 5), strides=(1, 1),
               padding='valid', activation='relu'),
        Dropout(0.25),
        Flatten(),
        Dense(nb_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

    # if logits:
    #     logits_tensor = model(input_ph)
    # # if logits:
    #     return model, logits_tensor
    # else:
    #     return model


def preprocess_transfer(X_train, y_train, X_test, y_test):
    # create two datasets one with digits below 5 and one with 5 and above
    X_train_lt5 = X_train[y_train < 5]
    y_train_lt5 = y_train[y_train < 5]
    X_test_lt5 = X_test[y_test < 5]
    y_test_lt5 = y_test[y_test < 5]

    X_train_gte5 = X_train[y_train >= 5]
    y_train_gte5 = y_train[y_train >= 5] - 5  # make classes start at 0 for
    X_test_gte5 = X_test[y_test >= 5]         # np_utils.to_categorical
    y_test_gte5 = y_test[y_test >= 5] - 5


def transfer_learning(data_shape=(28, 28, 1), nb_classes=5):
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = 2
    # convolution kernel size
    kernel_size = 3

    input_shape = data_shape


    # define two groups of layers: feature (convolutions) and classification (dense)
    feature_layers = [
        Conv2D(nb_filters, kernel_size, kernel_size,
                      padding='valid',
                      input_shape=input_shape),
        Activation('relu'),
        Conv2D(nb_filters, kernel_size, kernel_size),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Dropout(0.25),
        Flatten(),
    ]
    classification_layers = [
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ]

    # create complete model
    model = Sequential(feature_layers + classification_layers)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model
    # train model with lt5
    # freeze feature layers and rebuild model
    #for l in feature_layers:
    #    l.trainable = False

    # train model with gt5


def identity_model(test=None, inpt=Input(shape=(28, 28, 1)),
                   nb_filters=64, nb_classes=10):
    #x1 = Dropout(0.2)(inpt)
    x = Conv2D(32, 3, 3, activation='relu')(inpt)
    x = Conv2D(32, 3, 3, activation='relu')(x)
    x = BatchNormalization(axis=3)(x)

    c1 = MaxPooling2D((3, 3), strides=(1, 1))(x)
    c2 = Conv2D(96, 3, 3, activation='relu')(x)

    m1 = merge([c1, c2], mode='concat', concat_axis=3)
    m1 = Dropout(0.3)(m1)

    c1 = Conv2D(64, 1, 1, activation='relu', padding='same')(m1)
    c1 = Conv2D(96, 3, 3, activation='relu')(c1)

    c2 = Conv2D(64, 1, 1, activation='relu', padding='same')(m1)
    c2 = Conv2D(64, 7, 1, activation='relu', padding='same')(c2)
    c2 = Conv2D(64, 1, 7, activation='relu', padding='same')(c2)
    c2 = Conv2D(96, 3, 3, activation='relu', padding='valid')(c2)
    c2 = BatchNormalization(axis=3)(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=3)
    m2 = Dropout(0.3)(m2)

    p1 = MaxPooling2D((3, 3), strides=(2, 2), )(m2)
    p2 = Conv2D(192, 3, 3, activation='relu', strides=(2, 2))(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=3)
    m3 = BatchNormalization(axis=3)(m3)
    m3 = Activation('relu')(m3)
    x = Dropout(0.2)(m3)
    x = Flatten()(m3)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inpt, x, name='identity')
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def cnn(data_shape, nb_classes=10):
    model = Sequential([
        Dropout(0.5, input_shape=data_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=data_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        # MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(nb_classes, activation='softmax')
    ])

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, 5, 5, padding='same',
                          activation='relu', init='glorot_normal'))

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, 5, 5, padding='same',
                          activation='relu', init='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Conv2D(1, 2, 2, padding='same',
                          activation='tanh', init='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(inputs=[latent, image_class], outputs=fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, 3, padding='same', strides=(2, 2),
                          input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, 3, padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, 3, padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, 3, padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(inputs=image, outputs=[fake, aux])


def acgan():
        # batch and latent size taken from the paper
    nb_epochs = 50
    batch_size = 100
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(inputs=[latent, image_class], outputs=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, 10, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

    print('\nTesting for epoch {}:'.format(epoch + 1))
    # evaluate the testing loss here

    # generate a new batch of noise
    noise = np.random.uniform(-1, 1, (nb_test, latent_size))

    # sample some labels from p_c and generate images from them
    sampled_labels = np.random.randint(0, 10, nb_test)
    generated_images = generator.predict(
        [noise, sampled_labels.reshape((-1, 1))], verbose=False)

    X = np.concatenate((X_test, generated_images))
    y = np.array([1] * nb_test + [0] * nb_test)
    aux_y = np.concatenate((y_test, sampled_labels), axis=0)

    # see if the discriminator can figure itself out...
    discriminator_test_loss = discriminator.evaluate(
        X, [y, aux_y], verbose=False)

    discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

    # make new noise
    noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
    sampled_labels = np.random.randint(0, 10, 2 * nb_test)

    trick = np.ones(2 * nb_test)

    generator_test_loss = combined.evaluate(
        [noise, sampled_labels.reshape((-1, 1))],
        [trick, sampled_labels], verbose=False)

    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

    # generate an epoch report on performance
    train_history['generator'].append(generator_train_loss)
    train_history['discriminator'].append(discriminator_train_loss)

    test_history['generator'].append(generator_test_loss)
    test_history['discriminator'].append(discriminator_test_loss)

    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
        'component', *discriminator.metrics_names))
    print('-' * 65)

    OW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
    print(ROW_FMT.format('generator (train)',
                         *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                         *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                         *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                         *test_history['discriminator'][-1]))

    # save weights every epoch
    generator.save_weights(
        'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
    discriminator.save_weights(
        'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

    # generate some digits to display
    noise = np.random.uniform(-1, 1, (100, latent_size))

    sampled_labels = np.array([
        [i] * 10 for i in range(10)
    ]).reshape(-1, 1)

    # get a batch to display
    generated_images = generator.predict(
        [noise, sampled_labels], verbose=0)

    # arrange them into a grid
    img = (np.concatenate([r.reshape(-1, 28)
                           for r in np.split(generated_images, 10)
                           ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

    Image.fromarray(img).save(
        'plot_epoch_{0:03d}_generated.png'.format(epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))


def mlp_lle(nb_classes=10):
    model = Sequential([
        Dense(784, activation='relu', input_shape=(784,)),
        Dropout(0.5),
        Dense(300, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(nb_classes, activation='softmax')
        ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    return model


def cnn_lle(data_shape, nb_classes=10):
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=data_shape),
        Conv2D(64, 3, activation='relu'),
        Conv2D(128, 3, activation='relu'),
        Conv2D(256, 3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(324, activation='relu'),
        Dropout(0.45),
        Dense(162, activation='relu'),
        Dropout(0.35),
        Dense(81, activation='relu'),
        # Dropout(0.25),
        Dense(nb_classes, activation='softmax'),
        ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=1e-3),
                  metrics=['accuracy'])

    return model


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, strides=(1, 1)):
    def f(input):
        conv = Conv2D(nb_filter, (nb_row, nb_col),
                      kernel_regularizer=l2(1e-4), strides=strides,
                      kernel_initializer="he_normal",
                      padding="same")(input)
        norm = BatchNormalization(axis=3)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, strides=(1, 1)):
    def f(input):
        norm = BatchNormalization(axis=3)(input)
        activation = Activation("relu")(norm)
        return Conv2D(nb_filter, (nb_row, nb_col),
                      kernel_regularizer=l2(1e-4), strides=strides,
                      kernel_initializer="he_normal",
                      padding="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, net_type, init_strides=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1,
                                 strides=init_strides)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual, net_type)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, net_type, init_strides=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3,
                              strides=init_strides)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual, net_type)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual, net_type):
    if net_type == 'plain':
        return residual

    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[1] / residual._keras_shape[1]
    stride_height = input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == input._keras_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # assert stride_width == 1 and stride_height == 1 and equal_channels,
    # 'only identity is supported'
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(residual._keras_shape[3],
                          (1, 1),
                          strides=(stride_width, stride_height),
                          kernel_initializer="he_normal",
                          padding="valid")(input)

    if net_type == 'resnet':
        return merge([shortcut, residual], mode="sum")

    if net_type == 'squared_resnet':
        squared_shortcut = SquareMulLayer()(shortcut)
        return merge([shortcut, squared_shortcut, residual], mode="sum")

    assert False, 'invalid net_type: %s' % net_type


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, net_type,
                    is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(nb_filters, net_type, init_strides)(input)
        return input

    return f


# http://arxiv.org/pdf/1512.03385v1.pdf
def resnet(data_shape, repetations=3, net_type='resnet'):
    """net_type: plain, resnet, squared_resnet."""
    model_name = '{}_repetation_{}'.format(net_type, repetations)
    print(model_name)

    input = Input(shape=data_shape)
    conv1 = _conv_bn_relu(nb_filter=16, nb_row=3, nb_col=3)(input)
    # feature map size (32, 32)

    # Build residual blocks..
    block_fn = _basic_block
    block1 = _residual_block(block_fn, 16, repetations,
                             net_type, is_first_layer=True)(conv1)
    # feature map size (16, 16)
    block2 = _residual_block(block_fn, 32, repetations,
                             net_type)(block1)
    # feature map size (8, 8)
    block3 = _residual_block(block_fn, 64, repetations,
                             net_type)(block2)

    block3_dropout = Dropout(0.25)(block3)
    post_block_norm = BatchNormalization(axis=3)(block3_dropout)
    post_blob_relu = Activation("relu")(post_block_norm)

    # Classifier block
    pool2 = GlobalAveragePooling2D()(post_blob_relu)
    dense = Dense(10, kernel_initializer="he_normal",
                  kernel_regularizer=l2(1e-4),
                  activation="softmax")(pool2)

    model = Model(inputs=input, outputs=dense)
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=['accuracy'])

    # return model, model_name
    return model


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(100, 2), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2.) * epsilon


def variational_ae(data_shape):
    original_dim = 784
    latent_dim = 2
    intermediate_dim = 256
    x = Input(shape=(data_shape))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    y = CustomVariationalLayer()([x, x_decoded_mean, z_mean, z_log_var])
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    return vae, encoder, generator


def conv_ae(data_shape):
    input_img = Input(shape=data_shape)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # x = Conv2D(3, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    decoded = Flatten()(decoded)
    decoded = Dense(10, activation='softmax')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
                        metrics=['accuracy'])

    return autoencoder
