import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# TranX has values from -1 and 1, and TrainY has 3 times the TrainX
# and some randomness

trX = np.linspace(-1,1,101)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, init='uniform', activation='linear'))

# Take input x
# Apply weight w and bias b
# then use a linear activation to produce output
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))

# Use the mean squared error as the loss and simple gradient descent as the optimizer
model.compile(optimizer='sgd', loss='mse')

# Feed in the data with the fit function
model.fit(trX, trY, nb_epoch=200, verbose=1)

# Now print the weight after training
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))

# Store the model for future use
print('Storing model as my_model.h5')
model.save_weights('my_model.h5')
# To load a stored model
# model.load_weights('my_model_weights.h5')
