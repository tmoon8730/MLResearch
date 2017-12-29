# Using the squeezenet library https://github.com/rcmalli/keras-squeezenet/

import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


model = SqueezeNet()

img = image.load_img('cat.jpg', target_size=(227,227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = decode_predictions(model.predict(x))

print('Predicted: ', preds)
