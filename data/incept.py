import keras
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, Flatten, merge, Dropout
from keras.optimizers import Adam, Adadelta, Nadam, SGD
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import Lambda
# import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.preprocessing import image
import numpy as np
import os

layer_name = 'avg_pool'
# layer_name = 'mixed10'
model = InceptionV3(weights='imagenet', include_top=True)


data_path = "./animals_selected/"
# data_path = "/Users/albertotestoni/Downloads/img_incept/"
results = np.zeros((len(os.listdir(data_path)), 2048))
i = 0
list_files = sorted(os.listdir(data_path))

for filename in list_files:
    print(filename)
    img_path = os.path.join(data_path, filename)
    img = image.load_img(img_path, target_size=(299, 299))
    #img = image.load_img(img_path, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x)
    intermediate_output = intermediate_output.flatten()
    intermediate_output = np.reshape(intermediate_output, (1, 2048))
    results[i] = intermediate_output
    if i % 10 == 0:
        print("progress: {}/{} ".format(i, len(os.listdir(data_path))))
    i += 1
    print(filename + " " + ' '.join(str(f) for f in intermediate_output[0]))


#print(results)
#np.savetxt("55_vision_animals_xxxx.txt", results, fmt="%f", delimiter=" ")
