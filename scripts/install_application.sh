#!/bin/bash

gdrive_download_large () {
  fileid=$1
  filename=$2
  confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$confirm&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
}

gdrive_download_small () {
  fileid=$1
  filename=$2
  wget --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O $filename
}

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
gdrive_download_small 1bSs0-zSWZP2cPH25CQZkVrX9pgEdrxpl resnet50.index
gdrive_download_large 1v0j4psCHLMLMMZTg4R74ASAR_dwrULlW resnet50.data-00001-of-00002
gdrive_download_small 1vPQG2Q84DN8dnReMRnF09X9ZkCAkdQ22 resnet50.data-00000-of-00002
gdrive_download_small 1wDyo9jVwxwqO2OpCIFpmgKNLj7Fd1WCg covidnet.index
gdrive_download_large 1ReHiskVQvuISJWHJjf7ne2zepmIyZiGP covidnet.data-00001-of-00002
gdrive_download_small 1SyZ-Y9_xHPrnZ2WzaNpiMOsqm_9rwqc7 covidnet.data-00000-of-00002

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
