import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.91):
            print("Reached 91% accuracy so cancelling training!")
            self.model.stop_training = True

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Pixel normalization
training_images = training_images / 255.0 # This notation applies the division to every element in the array.
test_images = test_images / 255.0 

model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)],
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Exploring the model output

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])


callback = myCallback()

model.fit(training_images, training_labels, epochs=45, callbacks=[callback])