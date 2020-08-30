import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from covid19.models.covidnet import COVIDNet, COVIDNetLayer, PEPX
from covid19.datasets.data import plot_images
# from covid19.explainability.gradcam import GradCAM
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix



from covid19.datasets import covidx

covidx.generate_data('/home/fruggeri/Desktop/Projects/COVIDx', '../data/covidx')
exit(-1)



##########################
# 0. Prepare environment #
##########################

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='Architecture to use (covidnet or resnet50)')
parser.add_argument('--data', default='data', type=str, help='Path where to load data from')
parser.add_argument('--models', default='models', type=str, help='Path where to save models')
parser.add_argument('--logs', default='logs', type=str, help='Path where to save logs for TensorBoard')
parser.add_argument('--results', default='results', type=str, help='Path where to save evaluation results')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--img_height', default=224, type=int, help='Height of input images')
parser.add_argument('--img_width', default=224, type=int, help='Width of input images')
parser.add_argument('--img_channels', default=3, type=int, help='Channels of input images')
parser.add_argument('--factor', default=0.7, type=float, help='Factor to reduce LR on plateau')
parser.add_argument('--patience', default=5, type=int, help='Patience (number of epochs to wait before reducing LR)')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate (LR)')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--continue_fit', action='store_true', help='If the model is loaded, it is trained for other epochs')
parser.add_argument('--no-data_augmentation', action='store_true', help='Do not use data augmentation')
parser.add_argument('--no-class_imbalance', action='store_true', help='Do not compensate class imbalance')
parser.add_argument('--no-pretraining', action='store_true', help='Do not use weights of pre-training for ResNet-50')
parser.add_argument('--retraining', default=1, type=int, help='Number of layers to re-train (only for ResNet-50)')
parser.add_argument('--finetuning', default=1, type=int, help='Number of layers to fine-tune (only for ResNet-50)')
parser.add_argument('--epochs_finetuning', default=10, type=int, help='Number of epochs for fine-tuning phase (only for ResNet-50)')

args = parser.parse_args()
if args.finetuning < args.retraining:
    raise ValueError('Number of fine-tuned layers must be >= number of re-trained ones.')

# build model name so that files are not overwritten for different experiments
model_name = args.model
model_name += '_no-augmentation' if args.no_data_augmentation else ''
model_name += '_no-imbalance' if args.no_class_imbalance else ''
model_name += '_no-pretraining' if args.no_pretraining is not None else ''
model_name += '_no-finetuning' if (not args.no_pretraining and args.finetuning <= 1) else ''
model_logs_dir = os.path.join(args.logs, model_name)
model_path = os.path.join(args.models, model_name + '.h5')

print('====================')
print('Environment preparation')
print('====================')

print('Settings:')
for k, v in vars(args).items():
    print('\t{} = {}'.format(k, v))
print('--------------------')

answer = ''
while answer not in ['y', 'n']:
        answer = input('Double-check the settings. Are you sure you want to continue [y/n]? ').lower()
if answer == 'n':
    exit()

for d in [args.data, args.models, args.logs, args.results]:
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

train_dir = os.path.join(args.data, 'train')
train_dirs = [os.path.join(args.data, 'train', x) for x in os.listdir(train_dir)]
test_dir = os.path.join(args.data, 'test')
test_dirs = [os.path.join(args.data, 'test', x) for x in os.listdir(test_dir)]
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
plt.savefig(os.path.join(args.results, 'dataset.png'))
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
test_data_gen = test_image_generator.flow_from_directory(batch_size=args.batch_size, directory=test_dir,
                                                         target_size=(args.img_height, args.img_width))
class_names = sorted(test_data_gen.class_indices, key=test_data_gen.class_indices.get)
n_classes = test_data_gen.num_classes

# train generator
if args.no_data_augmentation:
    train_image_generator = ImageDataGenerator(rescale=1 / 255)
else:
    train_image_generator = ImageDataGenerator(rescale=1/255, featurewise_center=False,
                                               featurewise_std_normalization=False, rotation_range=10,
                                               width_shift_range=0.1, height_shift_range=0.1,
                                               horizontal_flip=True, brightness_range=(0.9, 1.1),
                                               zoom_range=(0.85, 1.15), fill_mode='constant', cval=0.)
train_data_gen = train_image_generator.flow_from_directory(batch_size=args.batch_size, directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(args.img_height, args.img_width))

# plot one batch (for debugging)
images = next(train_data_gen)[0]
plot_images(images)


##################
# 3. Build model #
##################

print()
print('====================')
print('Model architecture')
print('====================')

if os.path.isfile(model_path):          # load trained model
    model = load_model(model_path, custom_objects={'PEPX': PEPX, 'COVIDNetLayer': COVIDNetLayer, 'COVIDNet': COVIDNet})
    loaded = True
else:
    if 'covidnet' in model_name:
        model = COVIDNet(input_shape=(args.img_height, args.img_width, args.img_channels), n_classes=n_classes)
    elif 'resnet50' in model_name:
        if args.no_pretraining:
            base_model = ResNet50(include_top=False, pooling='avg', weights=None,
                                  input_shape=(args.img_height, args.img_width, args.img_channels))
            classifier = Dense(n_classes)
            model = Sequential([base_model, classifier])
        else:
            base_model = ResNet50(include_top=False, pooling='avg', weights='imagenet',
                                  input_shape=(args.img_height, args.img_width, args.img_channels))
            base_model.trainable = False
            classifier = Dense(n_classes)
            model = Sequential([base_model, classifier])
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

if not loaded or args.continue_fit:
    optimizer = Adam(learning_rate=args.learning_rate)
    loss = CategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        ModelCheckpoint(filepath=model_path),
        TensorBoard(log_dir=model_logs_dir)
    ]

    if args.no_class_imbalance:
        class_weight = None
        print('No compensation for unbalanced dataset')
    else:
        print('Weights for class imbalance:')
        class_weight = np.zeros(n_classes)
        for k, v in train_data_gen.class_indices.items():
            nk = len(os.listdir(os.path.join(train_dir, k)))
            class_weight[v] = tot_train / (n_classes*nk)
            print(k, class_weight[v])

    history = model.fit(train_data_gen, epochs=args.epochs, callbacks=callbacks,
                        steps_per_epoch=tot_train // args.batch_size,
                        validation_data=test_data_gen, validation_steps=tot_test // args.batch_size,
                        class_weight=class_weight)

    model.save(model_path)
    acc_h = history.history['accuracy']
    val_acc_h = history.history['val_accuracy']
    loss_h = history.history['loss']
    val_loss_h = history.history['val_loss']
    tot_epochs = args.epochs

    # fine-tuning
    if 'resnet50' in model_name and not args.no_pretraining and args.finetuning > 1:
        print('Starting fine-tuning...')
        optimizer = Adam(learning_rate=args.learning_rate / 10)     # note /10... FINE-tuning!
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # unfreeze layers to be fine-tuned
        model.trainable = True                                      # unfreeze all
        for layer in model.layers[:-args.finetuning]:
            layer.trainable = False                                 # freeze layers not to be fine-tuned
        model.summary()

        history = model.fit(train_data_gen, epochs=args.epochs, callbacks=callbacks,
                            steps_per_epoch=tot_train // args.batch_size, validation_data=test_data_gen,
                            validation_steps=tot_test // args.batch_size, class_weight=class_weight)

        model.save(model_path)
        acc_h += history.history['accuracy']
        val_acc_h += history.history['val_accuracy']
        loss_h += history.history['loss']
        val_loss_h += history.history['val_loss']
        tot_epochs += args.epochs_finetuning

    epochs_range = range(tot_epochs)
    plt.figure()
    plt.plot(epochs_range, history.history['accuracy'], label='training')
    plt.plot(epochs_range, history.history['val_accuracy'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(args.results, model_name + '_accuracy.png'))

    plt.figure()
    plt.plot(epochs_range, history.history['loss'], label='training')
    plt.plot(epochs_range, history.history['val_loss'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(args.results, model_name + '_loss.png'))
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

probabilities = model.predict(test_data_gen, steps=tot_test // args.batch_size + 1)
predictions = np.argmax(probabilities, axis=1)

# confusion matrix
cm = confusion_matrix(test_data_gen.classes, predictions)
ticks = np.arange(n_classes)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(args.results, model_name + '_confusion_matrix.png'))
plt.show()

# precision, recall, f1-score, accuracy, etc.
cr = classification_report(test_data_gen.classes, predictions, target_names=class_names)
print('Classification report')
print(cr)
with open(os.path.join(args.results, model_name + '_report.txt'), mode='w') as f:
    f.write(cr)

# Grad-CAM on some COVID-19 cases
covid_dir = os.path.join(test_dir, 'COVID-19')
filenames = os.listdir(covid_dir)
idx_images = np.arange(len(filenames))
np.random.shuffle(idx_images)
idx_images = idx_images[:10]
for i, idx in enumerate(idx_images):
    gc = GradCAM(model, os.path.join(covid_dir, filenames[idx]))
    gc.generate_and_visualize_heatmap(os.path.join(args.results, '{}_grad_cam_{}.png'.format(model_name, i)))
