import cv2
import os
import numpy as np
from keras.layers import Conv2D,Activation,Add, Input,Layer
from keras.models import Model
import tensorflow as tf

class SubpixelConv2d(Layer):
    """It is a 2D sub-pixel up-sampling layer, usually be used
    Parameters
    ------------
    scale : int
        The up-scaling ratio, a wrong setting will lead to dimension size error.
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__
    """
    def __init__(self,  scale=2, **kwargs):
        super(SubpixelConv2d, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        X = tf.depth_to_space(inputs, self.scale)
        return X

    def compute_output_shape(self, input_shape):
        if int(input_shape[-1]) / (self.scale**2) % 1 != 0:
            raise Exception(
                "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"
            )
        n_out_channel = int(int(input_shape[-1]) / (self.scale**2))
        return (input_shape[0] , None,None, n_out_channel)


def my_model():
    t_image = Input(shape=(None, None, 1))
    n_filters = 4
    n_resblock = 8
    n = Conv2D( filters=n_filters, kernel_size = (3, 3), strides = (1, 1), activation='relu', padding='SAME', name='n64s1/c')(t_image)
    temp = n

    # B residual blocks
    for i in range(n_resblock):
        nn = Conv2D(filters=n_filters, kernel_size = (3, 3), strides = (1, 1), activation=None, padding='SAME', name='n64s1/c1/%s' % i)(n)
        nn = Activation('relu')(nn)
        nn = Conv2D(filters=n_filters, kernel_size = (3, 3), strides = (1, 1), activation=None, padding='SAME', name='n64s1/c2/%s' % i)(nn)
        nn = Add(name='b_residual_add/%s' % i)([n, nn])
        n = nn

    n = Conv2D(filters=n_filters, kernel_size = (3, 3), strides = (1, 1), activation=None, padding='SAME', name='n64s1/c/m')(n)
    n = Add( name='add3')([n, temp])
    # B residual blacks end

    n = Conv2D(filters=n_filters*4, kernel_size = (3, 3), strides = (1, 1), activation=None, padding='SAME', name='n256s1/1')(n)
    n = SubpixelConv2d( scale=2,name='pixelshufflerx2/1')(n)
    n= Activation('relu')(n)

    n = Conv2D( filters=1, kernel_size = (1, 1), strides = (1, 1), activation='tanh', padding='SAME', name='out')(n)
    return Model(inputs=t_image,outputs=n)

def my_imgProcess(img, mm):
    print('process image......')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    imgy = np.expand_dims(img[:,:,0:1]/127.5 - 1,axis=0)
    imgyp = mm.predict(imgy)
    imgyp = np.uint8((imgyp + 1) * 127.5)
    imgyp_shape = imgyp.shape
    imgr = cv2.resize(img, (imgyp_shape[2], imgyp_shape[1]))
    imgr[:,:,0] = imgyp[0,:,:,0]
    imgr = cv2.cvtColor(imgr, cv2.COLOR_YCrCb2BGR)
    print('process image end')
    return [imgr]

def my_show(process_result):
    cv2.imshow('Result', process_result[0])
    cv2.waitKey()

def my_save(save_path, origin_imgName, process_result):
    print('save result......')
    for one_list in process_result:
        one_path = os.path.join(save_path, 'mp_'+ origin_imgName + '.jpg')
        cv2.imwrite(one_path, one_list)
    print('save result end')
