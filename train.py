import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from covidnet import COVIDNet, COVIDNetLayer, PEPX
from data import balanced_flow_from_directory, plot_images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


##########################
# 0. Prepare environment #
##########################

# np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='Architecture to use (covidnet or resnet50)')
parser.add_argument('--data', default='data', type=str, help='Path where to load data from')
parser.add_argument('--pretraining_data', default='data_imagenet', type=str,
                    help='Path where to load data for pre-training from')
parser.add_argument('--models', default='models', type=str, help='Path where to save models')
parser.add_argument('--logs', default='logs', type=str, help='Path where to save logs for TensorBoard')
parser.add_argument('--results', default='results', type=str, help='Path where to save evaluation results')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--img_height', default=224, type=int, help='Height of input images')
parser.add_argument('--img_width', default=224, type=int, help='Width of input images')
parser.add_argument('--img_channels', default=3, type=int, help='Channels of input images')
parser.add_argument('--factor', default=0.7, type=float, help='Factor to reduce LR on plateau')
parser.add_argument('--patience', default=5, type=int, help='Patience (number of epochs to wait before reducing LR)')
parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate (LR)')
parser.add_argument('--epochs', default=22, type=int, help='Number of epochs')
parser.add_argument('--no-data_augmentation', action='store_true', help='Do not use data augmentation')
parser.add_argument('--no-finetuning', action='store_true', help='Do not fine tune')
parser.add_argument('--no-pretraining', action='store_true', help='Do not fine tune')
parser.add_argument('--no-rebalancing', action='store_true', help='Do not rebalance batches')
parser.add_argument('--continue_fit', action='store_true',
                    help='If the model is loaded, it is trained for other epochs')

args = parser.parse_args()

data_dir = args.data
data_pretrain_dir = args.pretraining_data
models_dir = args.models
logs_dir = args.logs
results_dir = args.results
batch_size = args.batch_size
img_height = args.img_height
img_width = args.img_width
img_channels = args.img_channels
factor = args.factor
patience = args.patience
learning_rate = args.learning_rate
epochs = args.epochs
finetuning = not args.no_finetuning
pretraining = not args.no_pretraining
data_augmentation = not args.no_data_augmentation
rebalancing = not args.no_rebalancing
continue_fit = args.continue_fit

# build model name so that files are not overwritten for different experiments
model_name = args.model
model_name += '_finetuning' if finetuning else '_no-finetuning'
model_name += ('_pretraining_' + data_pretrain_dir.replace('data_', '')) if pretraining else '_no-pretraining'
model_name += '_augmentation' if data_augmentation else '_no-augmentation'
model_name += '_rebalancing' if rebalancing else '_no-rebalancing'
model_logs_dir = os.path.join(logs_dir, model_name)

print('====================')
print('Environment preparation')
print('====================')

print('Settings:')
for k, v in vars(args).items():
    print('\t{} = {}'.format(k, v))
print('--------------------')

for d in [data_dir, models_dir, logs_dir, results_dir]:
    try:
        os.mkdir(d)
        print('Directory', d, 'created')
    except FileExistsError:
        print('Directory', d, 'already existing')


###################
# 1. Examine data #
###################

print()
print('====================')
print('Data stats')
print('====================')

train_dir = os.path.join(data_dir, 'train')
train_dirs = [os.path.join(data_dir, 'train', x) for x in os.listdir(train_dir)]
test_dir = os.path.join(data_dir, 'test')
test_dirs = [os.path.join(data_dir, 'test', x) for x in os.listdir(test_dir)]
class_names = [x for x in os.listdir(train_dir)]

num_train = [len(os.listdir(x)) for x in train_dirs]
num_test = [len(os.listdir(x)) for x in test_dirs]
tot_train = sum(num_train)
tot_test = sum(num_test)

for k, n_train in zip(class_names, num_train):
    print('Training {} images: {}'.format(k, n_train))
for k, n_test in zip(class_names, num_test):
    print('Test {} images: {}'.format(k, n_test))
print('--------------------')
print('Total training images:', tot_train)
print('Total test images:', tot_test)

plt.figure()
x = np.arange(len(class_names))
plt.bar(x, num_train, width=.25, tick_label=class_names, label='train')
plt.bar(x + .25, num_test, width=.25, tick_label=class_names, label='test')
plt.ylabel('Number of images ($log_{10}$)')
plt.yscale('log')
plt.legend()
plt.savefig(os.path.join(results_dir, 'dataset.png'))
plt.show()


###########################
# 2. Build input pipeline #
###########################

print()
print('====================')
print('Data preparation')
print('====================')

# test generator
test_image_generator = ImageDataGenerator(rescale=1 / 255)
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size, directory=test_dir,
                                                         target_size=(img_height, img_width))
class_names = sorted(test_data_gen.class_indices, key=test_data_gen.class_indices.get)
n_classes = test_data_gen.num_classes

# train generator
if data_augmentation:
    train_image_generator = ImageDataGenerator(rescale=1/255, width_shift_range=.15, height_shift_range=.15,
                                               rotation_range=30, horizontal_flip=True, zoom_range=.2,
                                               brightness_range=(.5, 1.5))
else:
    train_image_generator = ImageDataGenerator(rescale=1 / 255)
if rebalancing:
    train_data_gen = balanced_flow_from_directory(image_generator=train_image_generator, batch_size=batch_size,
                                                  class_names=class_names, directory=train_dir, shuffle=True,
                                                  target_size=(img_height, img_width))
else:
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,
                                                               target_size=(img_height, img_width))

# plot one batch (for debugging)
images, labels = next(train_data_gen)
plot_images(images)


##################
# 3. Build model #
##################

print()
print('====================')
print('Model architecture')
print('====================')

model_path = os.path.join(models_dir, model_name + '.h5')

if os.path.isfile(model_path):      # existing file -> load model
    if 'covidnet' in model_name:
        model = load_model(model_path, custom_objects={'PEPX': PEPX, 'COVIDNetLayer': COVIDNetLayer,
                                                       'COVIDNet': COVIDNet})
    elif 'resnet50' in model_name:
        model = load_model(model_path)
    else:
        raise ValueError('Invalid model name. Supported models: covidnet, resnet50.')
    loaded = True
else:                               # otherwise create model
    if 'covidnet' in model_name:
        model = COVIDNet(input_shape=(img_height, img_width, img_channels), n_classes=n_classes)
    elif 'resnet50' in model_name:
        model = Sequential(name=model_name)
        model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet',
                           input_shape=(img_height, img_width, img_channels)))
        model.trainable = False
        model.add(Dense(n_classes))
    else:
        raise ValueError('Invalid model name. Supported models: covidnet, resnet50.')
    loaded = False
model.summary()


##################
# 4. Train model #
##################

print()
print('====================')
print('Model training')
print('====================')

if not loaded or continue_fit:
    optimizer = 'adam'
    loss = CategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=factor, patience=patience),
        ModelCheckpoint(filepath=model_path),
        TensorBoard(log_dir=model_logs_dir)
    ]

    history = model.fit(train_data_gen, epochs=epochs, callbacks=callbacks, steps_per_epoch=tot_train // batch_size,
                        validation_data=test_data_gen, validation_steps=tot_test // batch_size)

    epochs_range = range(epochs)

    plt.figure()
    plt.plot(epochs_range, history.history['accuracy'], label='training')
    plt.plot(epochs_range, history.history['val_accuracy'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_dir, model_name + '_accuracy.png'))

    plt.figure()
    plt.plot(epochs_range, history.history['loss'], label='training')
    plt.plot(epochs_range, history.history['val_loss'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, model_name + '_loss.png'))
    plt.show()
else:
    print('Model loaded... no training is going to be done')
    print('Use the option --continue_fit if you want to continue fitting')


#################
# 5. Test model #
#################

print()
print('====================')
print('Model evaluation')
print('====================')

probabilities = model.predict(test_data_gen, steps=tot_test // batch_size + 1)
predictions = np.argmax(probabilities, axis=1)

# confusion matrix
cm = confusion_matrix(test_data_gen.classes, predictions)
ticks = np.arange(n_classes)
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.xticks(ticks, class_names)
plt.yticks(ticks, class_names, rotation='vertical')
plt.tick_params(axis='both', length=0, labelsize=10)
plt.xlabel('predictions')
plt.ylabel('ground truth')
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, cm[i, j], fontsize=10, ha='center', va='center')
plt.savefig(os.path.join(results_dir, model_name + '_confusion_matrix.png'))
plt.show()

# precision, recall, f1-score, accuracy, etc.
cr = classification_report(test_data_gen.classes, predictions, target_names=class_names)
print('Classification report')
print(cr)
with open(os.path.join(results_dir, model_name + '_report.txt'), mode='w') as f:
    f.write(cr)
