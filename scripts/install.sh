#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <models_path>"
  exit 1
fi

# install packages
conda create --name covid19-detection tensorflow==2.3.0 -y
conda activate covid19-detection
pip install covid19-detection
pip install gdown

# download models
models_path=$1
rm -rf "$models_path"
mkdir "$models_path"
cd "$models_path"
gdown --id 1bSs0-zSWZP2cPH25CQZkVrX9pgEdrxpl --output resnet50.index
gdown --id 1v0j4psCHLMLMMZTg4R74ASAR_dwrULlW --output resnet50.data-00001-of-00002
gdown --id 1vPQG2Q84DN8dnReMRnF09X9ZkCAkdQ22 --output resnet50.data-00000-of-00002
gdown --id 1wDyo9jVwxwqO2OpCIFpmgKNLj7Fd1WCg --output covidnet.index
gdown --id 1ReHiskVQvuISJWHJjf7ne2zepmIyZiGP --output covidnet.data-00001-of-00002
gdown --id 1SyZ-Y9_xHPrnZ2WzaNpiMOsqm_9rwqc7 --output covidnet.data-00000-of-00002

# download icon
icon_path="~/.local/share/icons/covid19-detector.jpeg"
wget https://raw.githubusercontent.com/franco-ruggeri/dd2424-covid19-detection/master/covid19/ui/qt_designer/images/logo.jpeg -O $icon_path

# create desktop entry
cd ~/.local/share/applications
filename="covid19-detector.desktop"
echo "[Desktop Entry]" > $filename
echo "Version=0.3.0" >> $filename
echo "Type=Application" >> $filename
echo "Terminal=false" >> $filename
echo "Exec=bash -i \"conda activate covid19-detection && covid19-detector $models_path\"" >> $filename
echo "Name=COVID-19 Detector" >> $filename
echo "Icon=$icon_path" >> $filename
