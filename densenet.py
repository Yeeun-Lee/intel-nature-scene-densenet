from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D, Concatenate, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation

from keras.regularizers import l2
from keras.backend import concatenate
import keras
import keras.backend as K
import tensorflow as tf

def DenseNet(num_class, nb_blocks = 4, nb_filters = 128, depth = 40, growth_rate = 12, compression = 1,
             input_shape = (150, 150), channel = 3, weight_decay = 1e-4,
             include_top = False):
    """
    num_class = number of classes of your label data,
    nb_blocks = num of stages(num of dense blocks),
    nb_filters = initial num of filters(compressed after)
    denpth = L, growth_rate = k,
    H(l) = composite function,
    compression = 1(default),
    input_shape = (150, 150)(default),
    channel = image's RGB channel,
    weight_decay = used in kernel_regularizer,
    include_top = based on keras DenseNet121, if it is true, it includes
    fully connected layer
    """
    global concat_axis
    if K.image_data_format() =='channels_last':
        concat_axis = -1
        img_input = Input(shape=(input_shape[0], input_shape[1], channel), name = 'data')
    else:
        concat_axis = 1
        img_input = Input(shape=(channel, input_shape[0], input_shape[1]), name = 'data')
    assert (depth - 4) % 3 == 0, "Depth must be 3N + 4"
    nb_layers = int((depth-4)/3)

    # initial Convolution
    x = ZeroPadding2D(padding=(3, 3), name = 'conv1_zeropadding')(img_input)
    x = Conv2D(128, kernel_size=(7, 7), strides=(2, 2), padding='same',
               use_bias=False, name='conv1', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=concat_axis, name='conv1_bn')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='pool1')(x)

    # Dense Connectivity

    # add Dense blocks and Transition layers
    for idx in range(nb_blocks-1):
        stage = idx +2 # start from 2 conv
        x = dense_block(x, stage, nb_layers, growth_rate = growth_rate,
                        weight_decay = weight_decay)
        x = transition_layer(x, stage, nb_filters, weight_decay)
        nb_filters = int(nb_filters*compression)
    final_stage = stage+1

    #last dense block
    x = dense_block(x, final_stage, nb_layers, growth_rate, weight_decay)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                           name='conv'+str(final_stage)+'_tr_bn')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_tr')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage)+'_cls')(x)
    if include_top:
        x = Dense(num_class, activation='softmax')(x)

    model = Model(input = img_input, output = x, name='densenet')
    return model

def conv_block(x, stage, branch, growth_rate, weight_decay):
    """
    bottleneck layer(1*1 conv), H(i):comosite function(BN, ReLU, 3*3 Conv)
    Concatenation [x, x1]

    :return: concatenated feature maps
    """
    conv_name = "conv"+str(stage)+"_"+str(branch)
    relu_name = "relu"+str(stage)+"_"+str(branch)

    # 1x1 Conv(Blottleneck)
    inter_ch = growth_rate*4
    x1 = BatchNormalization(axis=concat_axis, name=conv_name+"_x1_bn")(x)
    x1 = Activation('relu', name=relu_name+'_x1')(x1)
    x1 = Conv2D(inter_ch, kernel_size=(1, 1), name=conv_name+"_x1",
                use_bias=False, kernel_regularizer=l2(weight_decay))(x1)
    x1 = Dropout(0.5)(x1)

    # BN, ReLU, 3x3 Conv(Composite Func)
    x1 = BatchNormalization(axis=concat_axis, name=conv_name+'_x2_bn')(x1)
    x1 = Activation('relu', name=relu_name+'_x2')(x1)
    x1 = Conv2D(growth_rate, kernel_size=(3, 3), padding='same', name=conv_name+'_x2',
                kernel_regularizer=l2(weight_decay))(x1)
    x1 = Dropout(0.5)(x1)
    x = Concatenate(axis=concat_axis, name=conv_name+"_concat")([x, x1])
    return x

def dense_block(x, stage, nb_layers, growth_rate, weight_decay):

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(x, stage, branch, growth_rate, weight_decay)
    return x

def transition_layer(x, stage, nb_filters, weight_decay):
    conv_name = 'conv'+str(stage)+'_tr'
    relu_name = 'relu'+str(stage)+'_tr'
    pool_name = 'pool'+str(stage)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu', name=relu_name)(x)
    x = Conv2D(nb_filters, kernel_size=(1, 1), padding='same', name=conv_name,
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name)(x)
    return x
