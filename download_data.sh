#!/bin/bash

echo 'Downloading Edward Robert Criscuolo, Yabo Fu, Yao Hao, Zhendong Zhang & Deshan Yang. (2023).'
echo 'Lung CT Deformable Image Registration Validation Dataset (Version 1) [Dataset]. Zenodo.'
echo 'https://doi.org/10.5281/zenodo.8200423'
echo 'Small but high-quality CT dataset'

curl -LS 'https://zenodo.org/records/8200423/files/lung_landmarks.zip?download=1' -o 'lung_landmarks.zip'
unzip lung_landmarks.zip
