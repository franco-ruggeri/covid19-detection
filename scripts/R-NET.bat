@echo off
title R-NET package installer
echo This process will install all the necessary requirements to run CovidNET!

set /P c=Do you wish to continue[y/n]?
if /I "%c%" EQU "Y" goto :install
if /I "%c%" EQU "N" goto :exit_process

:install
pip install covid19-detection
pip install gdown
gdown --id 1bSs0-zSWZP2cPH25CQZkVrX9pgEdrxpl --output resnet50.index
gdown --id 1v0j4psCHLMLMMZTg4R74ASAR_dwrULlW --output resnet50.data-00001-of-00002
gdown --id 1vPQG2Q84DN8dnReMRnF09X9ZkCAkdQ22 --output resnet50.data-00000-of-00002
gdown --id 1wDyo9jVwxwqO2OpCIFpmgKNLj7Fd1WCg --output covidnet.index
gdown --id 1ReHiskVQvuISJWHJjf7ne2zepmIyZiGP --output covidnet.data-00001-of-00002
gdown --id 1SyZ-Y9_xHPrnZ2WzaNpiMOsqm_9rwqc7 --output covidnet.data-00000-of-00002
pause
exit

:exit_process
echo Thank you for using this!
pause