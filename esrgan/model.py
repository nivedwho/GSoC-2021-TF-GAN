from keras.models import Model
from keras.layers import Input, Add, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense
from keras.layers import Lambda, Dropout
import tensorflow as tf

def _conv_block(input, filters = 64, strides = 1, activation = True):
    h = Conv2D(filters, kernel_size=3, strides = strides, padding='same')(input)
    if activation:
        h = LeakyReLU(0.2)(h)
    return h

def _conv_block_d(input, filters, strides = 1, bn = True):
    h = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
    h = LeakyReLU(alpha=0.2)(h)
    if bn:
        h = BatchNormalization(momentum=0.8)(h)
    return h

def SubpixelConv2D(name, scale=2):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)


def dense_block(input):
    h1 = _conv_block(input)
    h1 = Concatenate()([input, h1])

    h2 = _conv_block(h1)
    h2 = Concatenate()([input, h1, h2])

    h3 = _conv_block(h2)
    h3 = Concatenate()([input, h1, h2, h3])

    h4 = _conv_block(h3)
    h4 = Concatenate()([input, h1, h2, h3, h4])  

    h5 = _conv_block(h4, activation = False)
    h5 = Lambda(lambda x: x * 0.2)(h5)

    h = Add()([h5, input])
    
    return h

def RRDB(input):
    h = dense_block(input)
    h = dense_block(h)
    h = dense_block(h)
    h = Lambda(lambda x:x * 0.2)(h)
    out = Add()([h, input])
    return out

def upsample(x, number):
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
    x = SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
    x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
    return x

def ESRGAN_G():
    lr_input = Input(shape=(None, None, 3))
    
    x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(lr_input)
    x_start = LeakyReLU(0.2)(x_start)

    x = RRDB(x_start)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * 0.2)(x)
    x = Add()([x, x_start])

    x = upsample(x, 1)
    x = upsample(x, 2)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    hr_output = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

    model = Model(inputs=lr_input, outputs=hr_output)
    return model


def ESRGAN_D(filters = 64):
    img = Input(shape = (None, None, 3))
    
    x = _conv_block_d(img, filters, bn=False)
    x = _conv_block_d(x, filters = filters, strides=2)
    x = _conv_block_d(x, filters = filters * 2)

    x = _conv_block_d(x, filters = filters * 2, strides=2)
    x = _conv_block_d(x, filters = filters * 4)
    
    x = _conv_block_d(x, filters = filters * 4, strides=2)
    x = _conv_block_d(x, filters = filters * 8)
    
    x = _conv_block_d(x, filters = filters * 8, strides=2)

    x = Dense(filters * 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(1)(x)

    model = Model(inputs = img, outputs = x)
    return model