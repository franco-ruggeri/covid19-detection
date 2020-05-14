import os
import numpy as np
import matplotlib.pyplot as plt
from covidnet import COVIDNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model


##########################
# 0. Prepare environment #
##########################

print('====================')
print('Environment preparation')
print('====================')

np.random.seed(1)

data_dir = 'data'
models_dir = 'models'
logs_dir = 'logs'

try:
    os.mkdir(models_dir)
    print('Directory', models_dir, 'created')
except FileExistsError:
    print('Directory', models_dir, 'already existing')
try:
    os.mkdir(logs_dir)
    print('Directory', logs_dir, 'created')
except FileExistsError:
    print('Directory', logs_dir, 'already existing')


###################
# 1. Examine data #
###################

print()
print('====================')
print('Data stats')
print('====================')

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
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
print('Data preparation')
print('====================')

batch_size = 50
img_height = 224
img_width = 224
img_channels = 3

train_image_generator = ImageDataGenerator(rescale=1/255, validation_split=.9)
validation_image_generator = ImageDataGenerator(rescale=1/255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,
                                                           target_size=(img_height, img_width), subset='training')
test_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=test_dir,
                                                               target_size=(img_height, img_width))

sample_training_images, label = next(train_data_gen)
plot_images(sample_training_images)


##################
# 3. Build model #
##################

print()
print('====================')
print('Model architecture')
print('====================')

# select model here (COVID-Net or ResNet50)
model_name = 'covid_net'
# model_name = 'res_net_50'

model_path = os.path.join(models_dir, model_name + '.h5')
n_classes = train_data_gen.num_classes

if os.path.isfile(model_path):
    model = load_model(model_path)
    trained = True
else:
    if model_name == 'covid_net':
        model = COVIDNet(input_shape=(img_height, img_width, img_channels), n_classes=n_classes)
        model.build()
    elif model_name == 'res_net_50':
        model = Sequential(name=model_name)
        model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet',
                           input_shape=(img_height, img_width, img_channels)))
        model.trainable = False
        model.add(Dense(n_classes))
    else:
        raise ValueError('Invalid model name. Supported models: covid_net, res_net_50.')
    trained = False
model.summary()
exit()


##################
# 4. Train model #
##################

print()
print('====================')
print('Model training')
print('====================')

model_logs_dir = os.path.join(logs_dir, model.name)
try:
    os.mkdir(model_logs_dir)
except FileExistsError:
    pass

if not trained:
    optimizer = 'adam'
    loss = CategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    factor = 0.7
    patience = 5
    learning_rate = 2e-5
    epochs = 2

    lr_decay = ReduceLROnPlateau(monitor='loss', factor=factor, patience=patience)
    checkpoint = ModelCheckpoint(model_path)
    tensorboard = TensorBoard(log_dir=model_logs_dir)
    callbacks = [lr_decay, checkpoint, tensorboard]

    history = model.fit(train_data_gen, epochs=epochs, callbacks=callbacks)

    acc = history.history['acc']
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

print()
print('====================')
print('Model evaluation')
print('====================')

test_loss, test_acc = model.evaluate(test_data_gen, verbose=2)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
