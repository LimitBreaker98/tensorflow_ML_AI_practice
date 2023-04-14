# A convolutional neural network adds convolutions (image filters) to the technique of using neural networks to recognize images.
# Pooling can also be used, which is a technique that reduces the size of the image by removing some of the pixels, according to some math function (i.e the max, min, avg of a subset of pixels).

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.91):
            print("\nReached 91% accuracy so cancelling training!")
            self.model.stop_training = True

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1) # Additional dimension is color channel, 1 because this is monochrome (gray scale).
test_images = test_images.reshape(10000, 28, 28, 1)

# Pixel normalization, same as before.
training_images = training_images / 255.0 # This notation applies the division to every element in the array.
test_images = test_images / 255.0 

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2), # Reduces the image size by a factor of 2 in each dimension. Total pixels reduced by a factor of 4.
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Exploring the model output

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])


callback = myCallback()

model.fit(training_images, training_labels, epochs=45, callbacks=[callback])

# Model summary
model.summary()