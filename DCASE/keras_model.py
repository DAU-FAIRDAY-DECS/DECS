########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras import backend as K
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Model
import tensorflow as tf

########################################################################
# keras model
########################################################################
def get_model(input_dim, lr):
    """
    define the keras model
    (128*64*32*16*8*16*32*64*128)
    """

    x = Input(shape=(input_dim,))

    h = Dense(128)(x)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(32)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(16)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(16)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(32)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(input_dim)(h)

    model = Model(inputs=x, outputs=h)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss='mean_squared_error')

    return model

#########################################################################

def load_model(file_path):
    return keras.models.load_model(file_path, compile=False)

def clear_session():
    K.clear_session()
    