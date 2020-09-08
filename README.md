# COVID-19 project
This project is part of the course DD2424 Deep Learning in Data Science at KTH. The goal is to train a classifier for COVID-19 detection from chest X-ray (CXR) images and boost it with explainability. More information can be found in the [report](docs/report.pdf).

## How to install the package
```
pip install covid19-detection
```

## Structure of the package

The covid19 package is composed of the following sub-packages:
- covid19.datasets: contains utilities for generating COVIDx, HAM10000 and for building an input pipeline with tf.data.
- covid19.explainers: contains Grad-CAM and IG, two explainable AI methods, with some utilities for plotting the explanations.
- covid19.layers: contains layers used by models in covid19.models.
- covid19.metrics: contains utilities for computing and plotting metrics.
- covid19.models: contains ResNet50 and COVID-Net, two deep convolutional neural networks.

## Example of usage of the package

```
from covid19.models import COVID-Net
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_learning_curves

train_ds, train_ds_info = image_dataset_from_directory('path/to/COVIDx/train')
train_ds, _ = image_dataset_from_directory('path/to/COVIDx/validation')
model = COVID-Net(train_ds_info['n_classes'])
history = model.compile_and_fit(1e-4, 'cross_entropy', ['accuracy'], train_ds, val_ds, 30, 0, [])
model.save_weights('path/to/save/model)
plot_learning_curves(history, save_path='path/to/save/learning_curves)
```

## Scripts

The following scripts are present in this repository:
- generate_dataset: generate a dataset supported in covid19.datasets with training, validation and test splits.
- examine_dataset: generate a plot with the data distribution for a dataset supported in covid19.datasets.
- train_from_scratch: train ResNet50 or COVID-Net from covid19.models from scratch.
- train_transfer_learning: train ResNet50 or COVID-Net from covid19.models using transfer learning from another dataset.
- extract_pretrained: convert a model trained on a dataset to a pretrained model for another dataset.
- test: test the performance or the explainability of ResNet50 or COVID-Net from covid19.models.

A detailed description of the arguments and options of these scripts can be obtained with:
```
python <script>.py -h
```

## Example of usage of the scripts

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
