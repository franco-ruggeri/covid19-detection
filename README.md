# Explainable Detection of COVID-19 from Chest X-Ray Images
This project is part of the course DD2424 Deep Learning in Data Science at KTH. The goal is to train a classifier for COVID-19 detection from chest X-ray (CXR) images and boost it with explainability. More information can be found in the [report](https://github.com/franco-ruggeri/dd2424-covid19-detection/blob/master/docs/report.pdf). Also check out our [presentation](https://www.youtube.com/watch?v=c1TNhvAmddE&feature=youtu.be).

This package provides:
- An [application](#2-application) with a graphical user interface (GUI). This application can be used to make predictions on your images using trained models.
- A [suite of tools](#2-command-line-suite) with a command-line interface (CLI). These tools can be used to train and test new models.
- Several modules with a Keras-like API. These modules can be used in Python code.
  
# 1. Setup
For a complete installation, including the best models we trained, [download conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and run the following commands:
```
wget https://github.com/franco-ruggeri/dd2424-covid19-detection/blob/master/scripts/install.sh
bash -i "install.sh <models_path>" 
```

The package is distributed on [PyPi](https://pypi.org/), so can also be installed with:
```
pip install covid19-detection
```
However, the application needs trained models that have to be [downloaded](https://drive.google.com/drive/folders/1x7_xh1xNcuvT8j29y7pTyk_3nrFHNZd2?usp=sharing) or [trained on your own](3-command-line-suite).

## 2. Application
If you have done the complete installation, the application is installed in your system and can be launched by searching it among the applications. Otherwise, you can launch it with:
```
covid19-detector <models_path>
```

## 3. Command-line suite
The command-line suite is available under the *covid19-detection* command, but includes many subcommands. For the list of subcommands, run:
```
covid19-detection --help
```

## 4. Structure of the package
The covid19 package is composed of the following sub-packages:
- covid19.datasets: contains utilities for generating COVIDx, HAM10000 and for building an input pipeline with tf.data.
- covid19.explainers: contains Grad-CAM and IG, two explainable AI methods, with some utilities for plotting the explanations.
- covid19.layers: contains layers used by models in covid19.models.
- covid19.metrics: contains utilities for computing and plotting metrics.
- covid19.models: contains ResNet50 and COVID-Net, two deep convolutional neural networks.
- covid19.gui: contains graphical user interface implemented with [Qt](https://www.qt.io/).
- covid19.cli: contains command-line interface.
