# Explainable Detection of COVID-19 from Chest X-Ray Images
This project is part of the course DD2424 Deep Learning in Data Science at KTH. The goal is to train a classifier for COVID-19 detection from chest X-ray (CXR) images and boost it with explainability. More information can be found in the [report](https://github.com/franco-ruggeri/dd2424-covid19-detection/blob/master/docs/report.pdf). Also check out our [presentation](https://www.youtube.com/watch?v=c1TNhvAmddE&feature=youtu.be).

This package provides:
- An [application](#2-application) with a graphical user interface (GUI). This application can be used to make predictions on your images using trained models.
- A [suite of tools](#2-command-line-suite) with a command-line interface (CLI). These tools can be used to train and test new models.
- Several modules with a Keras-like API. These modules can be used in Python code.

# 1. Setup
The recommended installation is the following:
```
wget https://raw.githubusercontent.com/franco-ruggeri/dd2424-covid19-detection/master/scripts/install.sh -O install.sh
bash -i install.sh
```
Following the prompt, you can get a ready-to-use installation that uses the [best models we trained](https://drive.google.com/drive/folders/1x7_xh1xNcuvT8j29y7pTyk_3nrFHNZd2?usp=sharing).

The package is distributed on [PyPi](https://pypi.org/), so can be installed also with:
```
pip install covid19-detection
```
However, in this case you have to provide the trained models to the application. You can decide either to download the [best models we trained](https://drive.google.com/drive/folders/1x7_xh1xNcuvT8j29y7pTyk_3nrFHNZd2?usp=sharing) or to train your own models with the [command-line tools](3-command-line-suite).

## 2. Application
If you have done the recommended installation, you can launch the application by searching it among the applications. Otherwise, you can launch it from the terminal:
```
covid19-detector
```

## 3. Command-line suite
The command-line suite is available under the *covid19-detection* command. It provides several subcommands. The list can be retrieved with:
```
covid19-detection -h
```

More information about each subcommand can be obtained with:
```
covid19-detection <subcommand> -h
```

## 4. Package
You can import the package in your Python code with:
```
import covid19
```

The covid19 package is composed of the following sub-packages:
- covid19.datasets: contains utilities for generating COVIDx, HAM10000 and for building an input pipeline with tf.data.
- covid19.models: contains ResNet50 and COVID-Net, two deep convolutional neural networks.
- covid19.explainers: contains Grad-CAM and IG, two explainable AI methods, with some utilities for plotting the explanations.
- covid19.layers: contains layers used by models in covid19.models.
- covid19.metrics: contains utilities for computing and plotting metrics.
- covid19.gui: contains graphical user interface implemented with [Qt](https://www.qt.io/).
- covid19.cli: contains command-line interface.

Each subpackage provides interesting modules. For example, you can create a COVID-Net as follows:
```
from covid19.models import COVIDNet

model = COVIDNet(n_classes=3)
```

For more information about each class, see the comments.
