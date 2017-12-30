# Using the squeezenet library https://github.com/rcmalli/keras-squeezenet/

import numpy as np
# Keras
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
# Utilties
import sh
import os
import sys
# PythonImageLibrary
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

inputFile = sys.argv[1]

model = SqueezeNet()

img = image.load_img(inputFile, target_size=(227,227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = decode_predictions(model.predict(x))
most_likely = preds[0][0]
most_likely_object = most_likely[1]
most_likely_percent = most_likely[2]


fontFile = '/Users/tmoon/Code/MLResearch/fonts/abel-regular.ttf'

base = Image.open(inputFile).convert('RGBA')
txt = Image.new('RGBA', base.size, (255,255,255,0))

fnt = ImageFont.truetype(fontFile,40)

d = ImageDraw.Draw(txt)

d.text((0,0), most_likely_object + ' - {percent:.2%}'.format(percent=most_likely_percent), font=fnt, fill=(0,0,0,255))

out = Image.alpha_composite(base,txt)
out.show()

print '+++++++++++++++++++++++++++++++++++++++++++\n\n'
for results in preds:
    for result in results:
        print('Probability %0.2f%% => [%s]' % (100*result[2], result[1]))
print '\n\n+++++++++++++++++++++++++++++++++++++++++++'
