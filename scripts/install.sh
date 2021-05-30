#!/bin/bash

# create conda environment
echo -n "Do you want to create a conda environment ([Y]/n)? "; read
if [ -z "${REPLY}" ] || [ "${REPLY:0:1}" = "Y" ] || [ "${REPLY:0:1}" = "y" ]; then
  conda_env_default="covid19-detection"
  echo -n "Name of the conda environment [$conda_env_default]: "; read conda_env
  if [ -z "$conda_env" ]; then
    conda_env=$conda_env_default
  fi
  conda create --name $conda_env tensorflow==2.3.0 -y
  conda activate $conda_env
fi

# install packages
echo y | pip install covid19-detection gdown

# download models
models_path_default="$HOME/.covid19-detector/models"
echo -n "Path where to download the models [$models_path_default]: "; read models_path
if [ -z "$models_path" ]; then
    models_path=$models_path_default
fi
models_path=$(realpath "$models_path")
mkdir -p "$models_path"
cd "$models_path"
gdown --id 1bSs0-zSWZP2cPH25CQZkVrX9pgEdrxpl --output resnet50.index
gdown --id 1v0j4psCHLMLMMZTg4R74ASAR_dwrULlW --output resnet50.data-00001-of-00002
gdown --id 1vPQG2Q84DN8dnReMRnF09X9ZkCAkdQ22 --output resnet50.data-00000-of-00002
gdown --id 1wDyo9jVwxwqO2OpCIFpmgKNLj7Fd1WCg --output covidnet.index
gdown --id 1ReHiskVQvuISJWHJjf7ne2zepmIyZiGP --output covidnet.data-00001-of-00002
gdown --id 1SyZ-Y9_xHPrnZ2WzaNpiMOsqm_9rwqc7 --output covidnet.data-00000-of-00002

# download icon
icon_path="$HOME/.covid19-detector/icons"
mkdir -p $icon_path
icon_path=${icon_path}/covid19-detector.jpeg
wget -q https://raw.githubusercontent.com/franco-ruggeri/dd2424-covid19-detection/master/covid19/ui/qt_designer/images/logo.jpeg -O "$icon_path"

# create desktop entry
if [ -n "$conda_env" ]; then
  exec_cmd="conda activate covid19-detection && covid19-detector \"$models_path\""
else
  exec_cmd="covid19-detector \"$models_path\""
fi
filename="$HOME/.local/share/applications/covid19-detector.desktop"
echo "[Desktop Entry]" > $filename
echo "Version=0.3.0" >> $filename
echo "Type=Application" >> $filename
echo "Terminal=false" >> $filename
echo "Exec=bash -i -c '$exec_cmd'" >> $filename
echo "Name=COVID-19 Detector" >> $filename
echo "Icon=$icon_path" >> $filename

# create configurations
config_path="$HOME/.config/kth/covid19-detector.conf"
echo "[General]" > $config_path
echo "models_path=$models_path" >> $config_path
