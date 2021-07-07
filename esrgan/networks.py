import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Add, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, Conv2DTranspose
from keras.layers import Lambda, Dropout


"""
Generator (ESRGAN_G()) and Discriminator (ESRGAN_D()) models based on the architecture proposed in the 
paper "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks". 
"""

def _conv_block(input, filters = 32, strides = 1, activation = True):
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

    h = Add()([h5, input])
    
    return h

def RRDB(input):
    h = dense_block(input)
    h = dense_block(h)
    h = dense_block(h)
    h = Lambda(lambda x:x * 0.2)(h)
    out = Add()([h, input])
    return out


def upsample(x, number, use_bias = True):
    x = Conv2DTranspose(32, kernel_size= [3,3], strides=[2,2], padding='same', use_bias = use_bias)(x)
    return x

def ESRGAN_G(HParams, 
             num_filters = 64,
             out_channels = 3):
    """
    The Generator network for ESRGAN consisting of Residual in Residual Block as the 
    basic building unit. 

    Args : 
        num_filters : Number of num_filters for the convolutional layers used. 
        out_channels : Number of channels for the generated image. 
        use_bias : Whether to use bias or not for the convolutional layers. 
    
    Returns:
        The compiled model of the generator network where the inputs and outputs 
        of the model are defined as : 
            inputs -> Batch of tensors representing LR images.
            outputs -> Batch of generated HR images. 
    """
    lr_input = Input(shape=(HParams.hr_dimension//HParams.scale, HParams.hr_dimension//HParams.scale, 3))
    
    x = Conv2D(32, kernel_size= [3,3], strides=[1,1], padding='same', use_bias = True)(lr_input)
    x = LeakyReLU(0.2)(x)

    ref = x
    
    for i in range(HParams.trunk_size):
        x = RRDB(x)

    x = Conv2D(32, kernel_size= [3,3], strides=[1,1], padding='same', use_bias = True)(x)
    x = Add()([x, ref])

    x = upsample(x, 1)
    x = LeakyReLU(0.2)(x)

    x = upsample(x, 2)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(32, kernel_size= [3,3], strides=[1,1], padding='same', use_bias = True)(x)
    x = LeakyReLU(0.2)(x)

    hr_output = Conv2D(out_channels, kernel_size= [3,3], strides=[1,1], padding='same', use_bias = True)(x)

    model = Model(inputs=lr_input, outputs=hr_output)
    return model


def ESRGAN_D(num_filters = 64):
    """
    The discriminator network for ESRGAN. 

    Args :
        num_filters : Number of filters to be used in the first convolutional layer
    Returns : 
        The compiled model of the discriminator network where the inputs and outputs 
        of the model are defined as : 
            inputs -> Batch of tensors representing HR images.
            outputs -> Predictions for batch of input images. 
    """
    img = Input(shape = (None, None, 3))
    
    x = _conv_block_d(img, num_filters, bn=False)
    x = _conv_block_d(x, filters = num_filters, strides = strides)
    x = _conv_block_d(x, filters = num_filters * 2)
    x = _conv_block_d(x, filters = num_filters * 2, strides = strides)
    x = _conv_block_d(x, filters = num_filters * 4)
    x = _conv_block_d(x, filters = num_filters * 4, strides = strides)
    x = _conv_block_d(x, filters = num_filters * 8)    
    x = _conv_block_d(x, filters = num_filters * 8, strides = strides)

    x = Dense(num_filters * 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(1)(x)

    model = Model(inputs = img, outputs = x)
    return model
