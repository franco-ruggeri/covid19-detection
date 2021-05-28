#!/bin/bash

# get project path 
cd "$(dirname $0)/.."
project_dir=$(pwd)
project_dir=$(realpath "$project_dir")

# install packages
conda create --name covid19-detection tensorflow==2.3.0 -y
conda activate covid19-detection
pip install covid19-detection

# download models
models_dir=$project_dir/models
rm -rf "$models_dir"
mkdir "$models_dir"
cd "$models_dir"
wget -q "https://drive.google.com/uc?export=download&id=1bSs0-zSWZP2cPH25CQZkVrX9pgEdrxpl" -O resnet50.index
wget -q "https://drive.google.com/uc?export=download&id=1v0j4psCHLMLMMZTg4R74ASAR_dwrULlW" -O resnet50.data-00001-of-00002
wget -q "https://drive.google.com/uc?export=download&id=1vPQG2Q84DN8dnReMRnF09X9ZkCAkdQ22" -O resnet50.data-00000-of-00002
wget -q "https://drive.google.com/uc?export=download&id=1wDyo9jVwxwqO2OpCIFpmgKNLj7Fd1WCg" -O covidnet.index
wget -q "https://drive.google.com/uc?export=download&id=1ReHiskVQvuISJWHJjf7ne2zepmIyZiGP" -O covidnet.data-00001-of-00002
wget -q "https://drive.google.com/uc?export=download&id=1SyZ-Y9_xHPrnZ2WzaNpiMOsqm_9rwqc7" -O covidnet.data-00000-of-00002

# create desktop entry
cd ~/.local/share/applications
filename="covid19-detector.desktop"
echo "[Desktop Entry]" > $filename
echo "Version=1.0" >> $filename
echo "Type=Application" >> $filename
echo "Terminal=false" >> $filename
echo "Exec=bash -i \"$project_dir/scripts/run_application.sh\"" >> $filename
echo "Name=COVID-19 Detector" >> $filename
echo "Icon=$project_dir/covid19/ui/qt_designer/images/logo.jpeg" >> $filename
