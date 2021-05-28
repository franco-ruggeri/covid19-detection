#!/bin/bash

conda activate covid19-detection
python "$(dirname "$0")/run_application.py"
