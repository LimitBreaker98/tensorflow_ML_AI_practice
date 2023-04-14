from os.path import exists
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import urllib.request
import zipfile

url = 'https://storage.googleapis.com/learning-datasets/horse-or-human.zip'  
validation_url = 'https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip'

filename = 'horse-or-human.zip'
validation_filename = 'validation-horse-or-human.zip'

training_dir = 'horse-or-human/training'
validation_dir = 'horse-or-human/validation'

if not exists('horse-or-human/training'):
    # Download dataset and assign labels based on file structure.
    urllib.request.urlretrieve(url, filename)
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(training_dir)
    zip_ref.close()

if not exists('horse-or-human/validation'):
    # Download dataset and assign labels based on file structure.
    urllib.request.urlretrieve(validation_url, validation_filename)
    zip_ref = zipfile.ZipFile(validation_filename, 'r')
    zip_ref.extractall(validation_dir)
    zip_ref.close()

train_data_gen = ImageDataGenerator(rescale=1/255)
train_generator = train_data_gen.flow_from_directory(training_dir, target_size=(300, 300), batch_size=128, class_mode='binary') # categorical for more than 2 labels.

validation_data_gen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_data_gen.flow_from_directory(validation_dir, target_size=(300, 300), batch_size=32, class_mode='binary')

# Model construction

model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)), # Relu -> Rectified Linear Unit, an activation function which returns values only if they are positive.
    layers.MaxPooling2D(2,2), #Reduces the image size by a factor of 2 in each dimension. Uses the max function to determine the representative value of each subset of the pixels.
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid'), # the sigmoid function allows us to have a binary classification.
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

history = model.fit(train_generator, epochs=15, validation_data = validation_generator)