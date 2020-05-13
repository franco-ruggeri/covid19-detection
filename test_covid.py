import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


np.random.seed(1)

###################
# 1. Examine data #
###################

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
train_covid_dir = os.path.join(train_dir, 'COVID-19')
train_normal_dir = os.path.join(train_dir, 'normal')
train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')
test_covid_dir = os.path.join(test_dir, 'COVID-19')
test_normal_dir = os.path.join(test_dir, 'normal')
test_pneumonia_dir = os.path.join(test_dir, 'pneumonia')

num_covid_train = len(os.listdir(train_covid_dir))
num_normal_train = len(os.listdir(train_normal_dir))
num_pneumonia_train = len(os.listdir(train_pneumonia_dir))
num_covid_test = len(os.listdir(test_covid_dir))
num_normal_test = len(os.listdir(test_normal_dir))
num_pneumonia_test = len(os.listdir(test_pneumonia_dir))
total_train = num_covid_train + num_normal_train + num_pneumonia_train
total_test = num_covid_test + num_normal_test + num_pneumonia_test

print('====================')
print('Data stats')
print('====================')
print('Total training COVID-19 images:', num_covid_train)
print('Total training normal images:', num_normal_train)
print('Total training pneumonia images:', num_pneumonia_train)
print('Total test COVID-19 images:', num_covid_test)
print('Total test normal images:', num_normal_test)
print('Total test pneumonia images:', num_pneumonia_test)
print('--------------------')
print("Total training images:", total_train)
print("Total test images:", total_test)


###########################
# 2. Build input pipeline #
###########################

def plot_images(images_arr):
    fig, axes = plt.subplots(2, 4)
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


print()
print('====================')
print("Data preparation")
print('====================')

batch_size = 8
img_height = 244
img_width = 244
img_channels = 3

train_image_generator = ImageDataGenerator(rescale=1/255)
validation_image_generator = ImageDataGenerator(rescale=1/255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,
                                                           target_size=(img_height, img_width))
test_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=test_dir,
                                                               target_size=(img_height, img_width))

sample_training_images, label = next(train_data_gen)
plot_images(sample_training_images)


##################
# 3. Build model #
##################

print()
print('====================')
print("Model architecture")
print('====================')
resnet = Sequential()
resnet.add(ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, img_channels)))
resnet.trainable = False
resnet.add(GlobalAveragePooling2D())
resnet.add(Dense(3))
resnet.summary()


##################
# 4. Train model #
##################

print()
print('====================')
print("Model training")
print('====================')

optimizer = 'adam'
loss = CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy', Recall()]

resnet.compile(optimizer=optimizer, loss=loss, metrics=metrics)

factor = 0.7
patience = 5
learning_rate = 2e-5
epochs = 10
models_dir = 'models'
resnet_path = os.path.join(models_dir, resnet.name, '.h5')

lr_decay = ReduceLROnPlateau(monitor='train_loss', factor=factor, patience=patience)
checkpoint = ModelCheckpoint(resnet_path)
callbacks = [lr_decay, checkpoint]

history = resnet.fit(train_data_gen, epochs=epochs, callbacks=callbacks)

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochs)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='training accuracy')
plt.legend()
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='training loss')
plt.legend()
plt.title()
plt.show()


#################
# 5. Test model #
#################

