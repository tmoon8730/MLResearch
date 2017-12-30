# Using the squeezenet library https://github.com/rcmalli/keras-squeezenet/

import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import sh
import sys

inputFile = sys.argv[1]

model = SqueezeNet()

img = image.load_img(inputFile, target_size=(227,227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = decode_predictions(model.predict(x))[0]
most_likely = preds[0]
most_likely_object = most_likely[1]
most_likely_percent = most_likely[2]
print '+++++++++++++++++++++++++++++++++++++++++++\n\n'
print 'I am {percent:.2%} certain that image is a '.format(percent=most_likely_percent) + most_likely_object
sh.open(inputFile)
print '\n\n+++++++++++++++++++++++++++++++++++++++++++'
