#!/bin/bash
curl -L -o data/vehicle-insurance-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/imtkaggleteam/vehicle-insurance-data |\
  unzip data/vehicle-insurance-data.zip -d data/
