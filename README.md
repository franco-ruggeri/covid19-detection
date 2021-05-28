# Explainable Detection of COVID-19 from Chest X-Ray Images
This project is part of the course DD2424 Deep Learning in Data Science at KTH. The goal is to train a classifier for COVID-19 detection from chest X-ray (CXR) images and boost it with explainability. More information can be found in the [report](https://github.com/franco-ruggeri/dd2424-covid19-detection/blob/master/docs/report.pdf). Also check out our [presentation](https://www.youtube.com/watch?v=c1TNhvAmddE&feature=youtu.be).

This repository can be used in two different ways:
- As a [package](#1-python-package) in Python code, through a Keras-like API, using the [covid19](covid19) package. All the classes are documented.
- As a [standalone application](#2-standalone-application), using the [Qt application](scripts/run_application.py). New models can also be trained and tested using the [scripts](scripts). All the scripts provide documentation for the arguments.

## 1. Python package

### 1.1 Install
The package is distributed on [PyPi](https://pypi.org/), so can be installed with:
```
pip install covid19-detection
```

### 1.2 Structure of the package
The covid19 package is composed of the following sub-packages:
- covid19.datasets: contains utilities for generating COVIDx, HAM10000 and for building an input pipeline with tf.data.
- covid19.explainers: contains Grad-CAM and IG, two explainable AI methods, with some utilities for plotting the explanations.
- covid19.layers: contains layers used by models in covid19.models.
- covid19.metrics: contains utilities for computing and plotting metrics.
- covid19.models: contains ResNet50 and COVID-Net, two deep convolutional neural networks.
- covid19.ui: contains graphical user interface implemented with [Qt](https://www.qt.io/).

### 1.3 Example of usage
Here is a snippet of code to train a COVID-Net:
```
from covid19.models import COVIDNet
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_learning_curves

train_ds, train_ds_info = image_dataset_from_directory('path/to/COVIDx/train')
val_ds, _ = image_dataset_from_directory('path/to/COVIDx/validation')
model = COVIDNet(train_ds_info['n_classes'])
history = model.compile_and_fit(1e-4, 'cross_entropy', ['accuracy'], train_ds, val_ds, 30, 0, [])
model.save_weights('path/to/save/model')
plot_learning_curves(history, save_path='path/to/save/learning_curves)
```

## 2. Standalone application

### 2.1 Install on Linux
You need to execute the following commands:
```
git clone https://github.com/franco-ruggeri/dd2424-covid19-detection.git
cd dd2424-covid19-detection
bash -i scripts/install_application.sh
```

Now the application is installed in your system and can be launched by searching it among the applications. 

### 2.2 Install on Windows
TODO

### 2.3 Train and test models
After the installation, the Qt application uses the models in the *models* directory. You can train your own models and replace the default ones with them.

The following scripts allow training and testing your own ResNet50 and COVID-Net:
- generate_dataset: generates a dataset supported in covid19.datasets with training, validation and test splits.
- examine_dataset: generates a plot with the data distribution for a dataset supported in covid19.datasets.
- train_from_scratch: trains ResNet50 or COVID-Net from covid19.models from scratch.
- train_transfer_learning: trains ResNet50 or COVID-Net from covid19.models using transfer learning from another dataset.
- extract_pretrained: converts a model trained on a dataset to a pretrained model for another dataset.
- test: tests the performance or the explainability of ResNet50 or COVID-Net from covid19.models.

A detailed description of the arguments and options of these scripts can be obtained with:
```
python <script>.py -h
```

### 2.4 Example of usage
Here are some examples of usage of the scripts:
```
# pretrain ResNet50 on HAM10000
python train_from_scratch.py path/to/HAM10000 path/to/save/model_1 --architecture resnet50
python extract_pretrained.py path/to/HAM10000 path/to/COVIDx path/to/save/model_2 path/to/model_1 resnet50

# train on COVIDx with transfer learning from HAM10000
python train_transfer_learning.py path/to/COVIDx path/to/save/model_3 path/to/model_2

# test the model
python test.py performance path/to/COVIDx path/to/save/results path/to/model_3 resnet50
python test.py explainability path/to/COVIDx path/to/save/results path/to/model_3 resnet50 --explainer gradcam
python test.py explainability path/to/COVIDx path/to/save/results path/to/model_3 resnet50 --explainer ig
```
