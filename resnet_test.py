from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, merge, Input, Flatten, Dense

def relu(x):
    return Activation('relu')(x)

def neck(nip,nop,stride):
    def unit(x):
        nBottleneckPlane = int(nop / 4)
        nbp = nBottleneckPlane

        if nip==nop:
            ident = x

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,1,
            strides=(stride,stride))(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,3,padding='same')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,1)(x)

            out = merge([ident,x],mode='sum')
        else:
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            ident = x

            x = Conv2D(nbp,1,
            strides=(stride,stride))(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,3,padding='same')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,1)(x)

            ident = Conv2D(nop,1,
            strides=(stride,stride))(ident)

            out = merge([ident,x],mode='sum')

        return out
    return unit

def cake(nip,nop,layers,std):
    def unit(x):
        for i in range(layers):
            if i==0:
                x = neck(nip,nop,std)(x)
            else:
                x = neck(nop,nop,1)(x)
        return x
    return unit


def resnet(data_shape=(28, 28, 1)):
    inp = Input(shape=data_shape)
    x = inp
    
    x = Conv2D(16,3,padding='same')(x)
    
    x = cake(16,32,3,1)(x) #32x32
    x = cake(32,64,3,2)(x) #16x16
    x = cake(64,128,3,2)(x) #8x8
    
    x = BatchNormalization(axis=-1)(x)
    x = relu(x)
    
    x = AveragePooling2D(pool_size=(7,7),padding='valid')(x) #1x1
    x = Flatten()(x) # 128
    
    x = Dense(10)(x)
    x = Activation('softmax')(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
