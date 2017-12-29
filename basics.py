from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()

# This adds a Convolutional layer with 64 filters of size 3 * 3 to the graph
model.add(Conv2D(64,(3,3), activation='relu'))

# Fully connected layer called Dense, with the parameter
# specifying the number of outputs
model.add(Dense(256, activation='relu'))

# The input_shape specifies the shape of the input data
model.add(Conv2D(64,(3,3), activation='relu', input_shape(224,224,3)))

# After adding the layers then add the loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# Feed the data into the model
model.fit(x_train, y_train, batch_size=32, epochs=10,validation_data=(x_val, y_val))

# Finally evaluate the function
score = model.evaluate(x_test, y_test, batch_size=32)
