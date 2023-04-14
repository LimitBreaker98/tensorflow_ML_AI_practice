import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Sequential -> An object that represents multiple layers
# Dense -> The way neurons in each layer are connected.
    # Dense means every neuron in one layer is connected to every neuron in the next layer.
# Units -> Neurons in the layer.
# Input_shape -> The shape of the input data. 
# In this case, every x value is just a single value, so its shape is [1] for one dimensional.

layer_0 = Dense(units=1, input_shape=[1])
model = Sequential(layer_0)

# Stochastic Gradient Descent
model.compile(optimizer='sgd', loss='mean_squared_error')
# The optimizer will minimize the loss function over many iterations. 

xs = np.array([-1.0,0.0, 1.0,2.0,3.0,4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

# The optimizer takes the previous guess, the known values (xs and ys in input),
# and the loss function, to make a new guess for the fit function.
# It will do 500 iterations of this process to arrive at a function which fits the input and output.
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
print(f"Here is what I learned: {layer_0.get_weights()}")